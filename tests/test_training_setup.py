from unittest.mock import patch, sentinel

import pandas as pd
from PIL import Image
import pytest
import torch

from ai.dataset import CLASS_NAMES
from ai.models.model_factory import count_parameters, create_model
from ai.train_model import (
    EarlyStopping,
    build_class_weight_tensor,
    build_transforms,
    load_config,
    select_device,
)


def test_mobilenet_v2_baseline_architecture_without_downloading_weights():
    config = load_config()
    model = create_model(
        num_classes=len(config["classes"]),
        dropout=config["dropout"],
        pretrained=False,
        freeze_backbone=config["freeze_backbone"],
    )
    counts = count_parameters(model)

    assert config["pretrained_weights"] == "IMAGENET1K_V1"
    assert config["classes"] == CLASS_NAMES
    assert model.classifier[1].in_features == 1280
    assert model.classifier[1].out_features == 3
    assert counts == {"total": 2_227_715, "trainable": 3_843, "frozen": 2_223_872}
    assert config["device"] == "cuda"
    assert config["mixed_precision"] is True
    assert config["pin_memory"] is True
    assert config["num_workers"] == 2
    assert config["batch_size"] == 32


def test_validation_and_test_transforms_are_deterministic():
    transform_by_split = build_transforms(load_config())
    image = Image.new("RGB", (300, 260), color=(120, 180, 60))

    validation_first = transform_by_split["validation"](image)
    validation_second = transform_by_split["validation"](image)
    test_output = transform_by_split["test"](image)
    assert torch.equal(validation_first, validation_second)
    assert torch.equal(validation_first, test_output)


def test_baseline_augmentation_preserves_colour_semantics_and_oversampling_is_off():
    config = load_config()
    augmentation = config["train_augmentation"]
    assert augmentation["saturation"] == 0.0
    assert augmentation["hue"] == 0.0
    assert augmentation["brightness"] <= 0.08
    assert augmentation["contrast"] <= 0.08
    assert config["use_class_weights"] is True
    assert config["oversampling"] is False


def test_early_stopping_tracks_improvement_and_patience():
    stopping = EarlyStopping(patience=2, min_delta=0.01)
    assert stopping.update(1.0) == (True, False)
    assert stopping.update(0.995) == (False, False)
    assert stopping.update(0.994) == (False, True)


@patch("ai.train_model.torch.cuda.device_count", return_value=1)
@patch("ai.train_model.torch.cuda.is_available", return_value=True)
def test_cuda_device_mode_requires_and_selects_cuda(mock_available, mock_count):
    assert select_device("cuda") == torch.device("cuda:0")


@patch("ai.train_model.torch.cuda.device_count", return_value=0)
@patch("ai.train_model.torch.cuda.is_available", return_value=False)
def test_cuda_device_mode_stops_when_cuda_is_unavailable(mock_available, mock_count):
    with pytest.raises(RuntimeError, match="will not fall back to CPU"):
        select_device("cuda")


@patch("ai.train_model.torch.cuda.is_available", return_value=True)
def test_cpu_device_mode_forces_cpu_even_when_cuda_is_available(mock_available):
    assert select_device("cpu") == torch.device("cpu")


@pytest.mark.parametrize(
    ("available", "count", "expected"),
    [(True, 1, "cuda:0"), (False, 0, "cpu")],
)
def test_auto_device_mode_selects_available_device(available, count, expected):
    with patch("ai.train_model.torch.cuda.is_available", return_value=available), patch(
        "ai.train_model.torch.cuda.device_count", return_value=count
    ):
        assert select_device("auto") == torch.device(expected)


def test_class_weight_tensor_is_created_on_selected_device():
    manifest = pd.DataFrame({
        "class_name": ["Fresh", "Fresh", "Unripe", "Rotten"],
        "split": ["train", "train", "train", "train"],
    })
    config = {"use_class_weights": True}
    selected_device = torch.device("cuda:0")
    with patch("ai.train_model.torch.tensor", return_value=sentinel.class_weight_tensor) as tensor:
        weights, result = build_class_weight_tensor(manifest, config, selected_device)

    assert result is sentinel.class_weight_tensor
    assert set(weights) == {"Fresh", "Unripe", "Rotten"}
    assert tensor.call_args.kwargs["device"] == selected_device
