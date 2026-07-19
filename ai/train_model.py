import json
import multiprocessing
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms

from ai.dataset import (
    CLASS_NAMES,
    ManifestImageDataset,
    calculate_class_weights,
    load_manifest,
)
from ai.models.model_factory import count_parameters, create_model
from config.paths import BASE_DIR


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SUPPORTED_DEVICE_SETTINGS = {"cuda", "cpu", "auto"}


def load_config(config_path=None):
    path = Path(config_path).resolve() if config_path else BASE_DIR / "config" / "model_config.json"
    with path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    if config["model_name"] != "mobilenet_v2":
        raise ValueError("Training configuration must use mobilenet_v2.")
    if config["pretrained_weights"] != "IMAGENET1K_V1":
        raise ValueError("The baseline must use MobileNetV2 IMAGENET1K_V1 weights.")
    if config["classes"] != CLASS_NAMES:
        raise ValueError(f"Class order must be exactly: {CLASS_NAMES}")
    if config.get("oversampling", False):
        raise ValueError("Oversampling is not approved for the baseline training stage.")
    if config.get("device") not in SUPPORTED_DEVICE_SETTINGS:
        raise ValueError("Device must be one of: cuda, cpu, auto.")
    if not isinstance(config.get("mixed_precision"), bool):
        raise ValueError("mixed_precision must be true or false.")
    if not isinstance(config.get("pin_memory"), bool):
        raise ValueError("pin_memory must be true or false.")
    if not isinstance(config.get("num_workers"), int) or config["num_workers"] < 0:
        raise ValueError("num_workers must be a non-negative integer.")
    return config


def resolve_project_path(path_value):
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (BASE_DIR / path).resolve()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_setting):
    setting = str(device_setting).lower()
    if setting not in SUPPORTED_DEVICE_SETTINGS:
        raise ValueError("Device must be one of: cuda, cpu, auto.")
    if setting == "cpu":
        return torch.device("cpu")
    if setting == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError(
                "Training requires CUDA because device='cuda', but CUDA is unavailable. "
                "FreshSight will not fall back to CPU. Run scripts\\check_cuda.ps1 and fix the CUDA environment."
            )
        return torch.device("cuda:0")
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return torch.device("cuda:0")
    return torch.device("cpu")


def configure_backend(device):
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_cuda_details(device):
    if device.type != "cuda":
        return None
    properties = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "pytorch_version": torch.__version__,
        "cuda_build_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(device),
        "total_memory_bytes": int(properties.total_memory),
        "free_memory_bytes": int(free_bytes),
        "runtime_total_memory_bytes": int(total_bytes),
    }


def build_transforms(config):
    augmentation = config["train_augmentation"]
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            config["image_size"],
            scale=tuple(augmentation["random_resized_crop_scale"]),
            ratio=tuple(augmentation["random_resized_crop_ratio"]),
        ),
        transforms.RandomHorizontalFlip(p=augmentation["horizontal_flip_probability"]),
        transforms.RandomRotation(degrees=augmentation["rotation_degrees"]),
        transforms.ColorJitter(
            brightness=augmentation["brightness"],
            contrast=augmentation["contrast"],
            saturation=augmentation["saturation"],
            hue=augmentation["hue"],
        ),
        transforms.ToTensor(),
        normalize,
    ])
    evaluation_transform = transforms.Compose([
        transforms.Resize(config["resize_size"]),
        transforms.CenterCrop(config["image_size"]),
        transforms.ToTensor(),
        normalize,
    ])
    return {
        "train": train_transform,
        "validation": evaluation_transform,
        "test": evaluation_transform,
    }


def prepare_dataloaders(config, device):
    manifest_path = resolve_project_path(config["manifest_path"])
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest is missing: {manifest_path}")
    inspection_report = BASE_DIR / "evaluation" / "outputs" / "dataset_inspection.json"
    if inspection_report.is_file() and inspection_report.stat().st_mtime > manifest_path.stat().st_mtime:
        raise RuntimeError(
            "Dataset manifest is older than the latest inspection report and must not be reused. "
            "Complete duplicate cleanup, rerun inspection, and create a new split manifest."
        )
    manifest = load_manifest(manifest_path, validate=True)
    transform_by_split = build_transforms(config)
    generator = torch.Generator().manual_seed(config["seed"])
    use_pin_memory = bool(config["pin_memory"] and device.type == "cuda")
    persistent_workers = config["num_workers"] > 0
    if config["num_workers"] == 0:
        print("WARNING: num_workers=0; DataLoader multiprocessing is disabled for Windows fallback mode.")
    loaders = {}
    for split_name in ("train", "validation", "test"):
        dataset = ManifestImageDataset(
            manifest_path=manifest_path,
            split=split_name,
            transform=transform_by_split[split_name],
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=split_name == "train",
            num_workers=config["num_workers"],
            pin_memory=use_pin_memory,
            persistent_workers=persistent_workers,
            generator=generator if split_name == "train" else None,
        )
    return manifest, loaders


class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.bad_epochs = 0

    def update(self, validation_loss):
        improved = validation_loss < self.best_loss - self.min_delta
        if improved:
            self.best_loss = validation_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience


def build_class_weight_tensor(manifest, config, device):
    weights_by_class = calculate_class_weights(manifest, split="train")
    if not config["use_class_weights"]:
        return weights_by_class, None
    tensor = torch.tensor(
        [weights_by_class[name] for name in CLASS_NAMES],
        dtype=torch.float32,
        device=device,
    )
    return weights_by_class, tensor


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, mixed_precision=False):
    training = optimizer is not None
    model.train(training)
    if training and not any(parameter.requires_grad for parameter in model.features.parameters()):
        model.features.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    non_blocking = device.type == "cuda"
    amp_enabled = bool(mixed_precision and device.type == "cuda")
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_samples += batch_size
    if total_samples == 0:
        raise ValueError("A training or validation loader is empty.")
    return total_loss / total_samples, total_correct / total_samples


def save_checkpoint(path, model, optimizer, epoch, config, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "classes": CLASS_NAMES,
        "config": config,
        "metrics": metrics,
    }, path)


def save_history(history, json_path, csv_path):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as history_file:
        json.dump(history, history_file, indent=2)
    pd.DataFrame(history).to_csv(csv_path, index=False)


def main():
    config = load_config()
    seed_everything(config["seed"])
    device = select_device(config["device"])
    configure_backend(device)
    cuda_details = get_cuda_details(device)
    print("=== FreshSight CUDA / Device Selection ===")
    print(f"Configured device mode: {config['device']}")
    print(f"Selected device: {device}")
    if cuda_details:
        print(f"PyTorch version: {cuda_details['pytorch_version']}")
        print(f"CUDA build version: {cuda_details['cuda_build_version']}")
        print(f"CUDA device count: {cuda_details['device_count']}")
        print(f"GPU model: {cuda_details['gpu_name']}")
        print(f"Total VRAM: {cuda_details['total_memory_bytes'] / 1024**3:.2f} GiB")
        print(f"Available VRAM: {cuda_details['free_memory_bytes'] / 1024**3:.2f} GiB")
        torch.cuda.empty_cache()
    manifest, loaders = prepare_dataloaders(config, device)

    try:
        model = create_model(
            num_classes=len(config["classes"]),
            dropout=config["dropout"],
            pretrained=True,
            freeze_backbone=config["freeze_backbone"],
        ).to(device)
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "CUDA ran out of memory while loading the model. Keep automatic batch-size changes disabled "
            "and reduce batch_size from 32 to 16 before retrying."
        ) from exc
    parameter_counts = count_parameters(model)
    if device.type == "cuda":
        print("=== CUDA Memory After Model Loading ===")
        print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MiB")
        print(f"Reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MiB")
        print(f"Total VRAM: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GiB")

    weights_by_class, class_weight_tensor = build_class_weight_tensor(manifest, config, device)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["scheduler_factor"],
        patience=config["scheduler_patience"],
    )
    early_stopping = EarlyStopping(
        patience=config["early_stopping_patience"],
        min_delta=config["early_stopping_min_delta"],
    )
    amp_enabled = bool(config["mixed_precision"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_path = resolve_project_path(config["best_checkpoint_path"])
    last_path = resolve_project_path(config["last_checkpoint_path"])
    history_json = resolve_project_path(config["training_history_json"])
    history_csv = resolve_project_path(config["training_history_csv"])
    config_output = resolve_project_path(config["training_config_output"])
    config_output.parent.mkdir(parents=True, exist_ok=True)
    runtime_config = {
        **config,
        "resolved_device": str(device),
        "cuda_details": cuda_details,
        "mixed_precision_enabled": amp_enabled,
        "parameter_counts": parameter_counts,
        "class_weights": weights_by_class if config["use_class_weights"] else None,
        "split_counts": {
            name: int(count) for name, count in manifest["split"].value_counts().to_dict().items()
        },
    }
    with config_output.open("w", encoding="utf-8") as config_file:
        json.dump(runtime_config, config_file, indent=2)

    print("=== FreshSight MobileNetV2 Baseline Training ===")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Classes: {', '.join(CLASS_NAMES)}")
    print(f"Manifest: {resolve_project_path(config['manifest_path'])}")
    print(f"Samples: train={len(loaders['train'].dataset)}, validation={len(loaders['validation'].dataset)}, test={len(loaders['test'].dataset)}")
    print(f"Parameters: total={parameter_counts['total']}, trainable={parameter_counts['trainable']}, frozen={parameter_counts['frozen']}")
    print(f"Class weights enabled: {config['use_class_weights']}")
    print(f"Mixed precision enabled: {amp_enabled}")
    print(f"pin_memory: {loaders['train'].pin_memory}")
    print(f"num_workers: {config['num_workers']}")
    print(f"persistent_workers: {loaders['train'].persistent_workers}")
    if config["use_class_weights"]:
        print("Class weights: " + ", ".join(f"{name}={weights_by_class[name]:.6f}" for name in CLASS_NAMES))
    print("The test split is held out and is not evaluated during training.")

    history = []
    started_at = time.time()
    for epoch in range(1, config["num_epochs"] + 1):
        try:
            train_loss, train_accuracy = run_epoch(
                model,
                loaders["train"],
                criterion,
                device,
                optimizer,
                scaler=scaler,
                mixed_precision=amp_enabled,
            )
            validation_loss, validation_accuracy = run_epoch(
                model,
                loaders["validation"],
                criterion,
                device,
                optimizer=None,
                scaler=None,
                mixed_precision=amp_enabled,
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"CUDA out of memory with batch_size={config['batch_size']}. Training stopped; "
                "reduce batch_size to 16 and retry. The batch size was not changed automatically."
            ) from exc
        except RuntimeError as exc:
            if config["num_workers"] > 0 and "DataLoader worker" in str(exc):
                raise RuntimeError(
                    "A Windows DataLoader worker failed. Training stopped without changing configuration. "
                    "Set num_workers to 0 explicitly and retry."
                ) from exc
            raise
        scheduler.step(validation_loss)
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.time() - started_at,
        }
        history.append(metrics)
        improved, should_stop = early_stopping.update(validation_loss)
        if improved:
            save_checkpoint(best_path, model, optimizer, epoch, runtime_config, metrics)
        save_checkpoint(last_path, model, optimizer, epoch, runtime_config, metrics)
        save_history(history, history_json, history_csv)
        print(
            f"Epoch {epoch:02d}/{config['num_epochs']} | "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.2%} | "
            f"val_loss={validation_loss:.4f} val_acc={validation_accuracy:.2%} | "
            f"lr={optimizer.param_groups[0]['lr']:.6g} | "
            f"best={'yes' if improved else 'no'}"
        )
        if should_stop:
            print(f"Early stopping triggered after {early_stopping.bad_epochs} epochs without improvement.")
            break

    print("=== Training Outputs ===")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"History JSON: {history_json}")
    print(f"History CSV: {history_csv}")
    print(f"Runtime configuration: {config_output}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
