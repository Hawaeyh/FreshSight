# FreshSight

FreshSight is a local papaya freshness assessment system combining a primary MobileNetV2 classifier with supporting MATLAB image processing.

## Current classes

- Fresh
- Unripe
- Rotten

Class-folder matching is case-insensitive. Labels are normalized internally to the names above. Folders such as `semi_fresh`, `Semi-Fresh`, `unripen`, and `UnRippen` are unsupported and excluded.

## Prerequisites

- Windows 10
- Python 3.11 available through `py -3.11` or `python`
- MATLAB R2025b
- MATLAB Engine for Python (needed for MATLAB analysis, but not for dataset inspection)

The release-matched MATLAB Engine package is installed separately from `requirements.txt` as `matlabengine==25.2.2`.

## Initial environment setup

From the project root, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_environment.ps1
```

This creates the project-local `.venv`, upgrades pip, and installs `requirements.txt`. The command changes execution policy only for that PowerShell process; it does not modify the global policy.

## Run FreshSight

```powershell
powershell -ExecutionPolicy Bypass -File start_freshsight.ps1
```

If `.venv` is missing, the launcher runs environment setup first. The local Flask address is `http://127.0.0.1:5000`.

The active model registry points to the evaluated epoch-19 cleaned-baseline checkpoint without copying it. MobileNetV2 loads lazily and is reused. MATLAB starts lazily and remains independently available for measurements, masks, highlighted damage, and rule comparison.

Additional local, read-only operational pages are available after startup:

- `/history` — analysis history with immutable original AI predictions;
- `/admin/feedback` — feedback review and ranked active-learning candidates;
- `/dashboard/performance` — existing held-out AI and MATLAB evaluation results;
- `/system-health` — checkpoint, CUDA, MATLAB-session, and database status.

Grad-CAM is generated on demand from an analysis-detail page and saved under `outputs/explainability/`. It performs a backward pass only for visualization and does not update optimizer state, checkpoint data, or model weights. Feedback approval and active-learning scoring are review workflows only; neither can trigger retraining.

## Enhanced MATLAB segmentation development test

The supporting MATLAB subsystem now exposes background removal, raw/clean/inner fruit masks, quality gating, exclusive colour masks, HSV/Lab channels, texture measurements, and boundary-safe damage masks. Poor segmentation suppresses measurements and cannot influence agreement or measurement-based recommendation text.

Run the controlled train/validation-only sample set with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/test_enhanced_matlab_segmentation.ps1
```

Inspect `outputs/matlab_segmentation_test/` visually before treating the new segmentation as validated. No held-out test image is permitted by the runner.

## Controlled MATLAB damage refinement

The prepared damage-evidence stage separates stem, smooth shadow, reflection, brown decay, dark decay, mold, lesion, and abnormal texture while consuming the frozen inner segmentation mask. Its 24 declared samples are balanced across Fresh, Unripe, and Rotten and validated against train/validation manifest rows only.

After reviewing the sample annotations, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/test_enhanced_matlab_damage.ps1
```

Results are written under `outputs/matlab_damage_test/`, including a before/after CSV for the original ten segmentation-development images. Visually inspect the evidence masks before claiming improvement.

## Inspect the dataset

```powershell
powershell -ExecutionPolicy Bypass -File inspect_dataset.ps1
```

The inspector reads `dataset/` by default. It does not move, copy, rename, delete, or split images. It reports:

- counts for Fresh, Unripe, and Rotten;
- unsupported folders and file extensions;
- corrupted or unreadable images;
- exact duplicate files using SHA-256;
- empty or missing required class folders;
- class imbalance using a max/min count ratio greater than `1.5`.

Reports are written locally to:

- `evaluation/outputs/dataset_inspection.json`
- `evaluation/outputs/dataset_inspection.csv`

An optional dataset path can be passed directly to the Python inspector after activation:

```powershell
python evaluation/inspect_dataset.py --dataset-path "path\to\dataset"
```

## Create the dataset manifest

After reviewing a successful inspection report, create a deterministic, duplicate-aware split manifest with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_dataset_split.ps1
```

The splitter safely recomputes SHA-256 values and keeps every exact-duplicate group in one indivisible split. It approximates the requested 70% training, 15% validation, and 15% test ratios independently for each normalized class. It validates that no hash crosses splits, every required class appears in every split, unsupported classes are absent, and source paths are unique.

No images are copied, deleted, renamed, or modified. Outputs are limited to:

- `evaluation/outputs/dataset_manifest.csv`
- `evaluation/outputs/dataset_split_summary.json`

The manifest columns are `source_path`, `class_name`, `class_index`, `split`, `sha256`, and `duplicate_group_id`. `ai.dataset.ManifestImageDataset` can load a selected split directly from this manifest.

If inspection detects class imbalance, the dataset module can calculate inverse-frequency class weights for a future training run. Splitting does not oversample or synthesize images.

## Preview exact-duplicate cleanup

After generating a current inspection report, preview the quarantine plan with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/preview_duplicate_cleanup.ps1
```

Preview mode recomputes every reported SHA-256 hash and writes only `evaluation/outputs/duplicate_cleanup_plan.csv` and `evaluation/outputs/duplicate_cleanup_summary.json`. It does not change dataset images or create the quarantine directory. Review both files before approving apply.

An approved apply moves redundant copies into class-preserving folders under `dataset_duplicates_backup/`, records moves in `evaluation/outputs/duplicate_cleanup_audit.csv`, and supports rollback. Apply and rollback launchers require explicit confirmation switches. After either operation, rerun dataset inspection and create a new split; older manifests and summaries must not be reused.

## MobileNetV2 baseline preparation

The prepared baseline reads images directly from `evaluation/outputs/dataset_manifest.csv`. It uses MobileNetV2 with ImageNet `IMAGENET1K_V1` weights, a dropout layer, and a three-output classifier ordered as Fresh, Unripe, and Rotten.

The initial configuration freezes the feature backbone and trains the replacement classifier. Training augmentation uses restrained crops, horizontal flips, rotations, brightness, and contrast adjustments; hue and saturation changes are disabled to preserve freshness-colour evidence. Validation and test preprocessing use deterministic resize and centre-crop operations.

Inverse-frequency class weights are enabled for the first baseline. Oversampling and synthetic images remain disabled. The test split is held out during training.

Training does not begin on import or during environment setup. After explicit approval, it can be started with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_training.ps1
```

The run saves separate best and last checkpoints, JSON/CSV history, and a runtime configuration snapshot. Model evaluation remains a later, separate stage.

### CUDA requirement for the cleaned baseline

The cleaned baseline is configured with `device: cuda`. It requires a CUDA-enabled PyTorch installation and will stop with an error rather than fall back to CPU. Verify CUDA before training:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/check_cuda.ps1
```

The baseline uses mixed precision, pinned DataLoader memory, two Windows worker processes, and cuDNN benchmarking for fixed-size inputs. Batch size remains 32. If CUDA runs out of memory, training stops and instructs the user to set batch size to 16; it never changes batch size silently. If Windows DataLoader workers fail, set `num_workers` to `0` explicitly and rerun so the fallback is visible in the runtime configuration.

## Manual activation

To activate `.venv` directly:

```powershell
.\.venv\Scripts\Activate.ps1
```

To use the helper and keep the environment active in the current PowerShell session, dot-source it:

```powershell
. .\scripts\activate_environment.ps1
```

## MATLAB Engine setup

After activating `.venv`, install the release-matched MATLAB Engine package for MATLAB R2025b:

```powershell
python -m pip install matlabengine==25.2.2
```

No project script changes the global PowerShell execution policy.

## Active MATLAB subsystem

The canonical rule-based implementation is under `matlab/`:

- `api/run_freshsight_api.m`
- `core/analyze_papaya.m`
- `preprocessing/preprocess_papaya_image.m`
- `segmentation/segment_papaya.m`
- `features/extract_papaya_features.m`
- `classification/classify_papaya_freshness.m`
- `visualization/create_damage_highlight.m`
- `scripts/run_single_image_demo.m`
- `scripts/evaluate_rule_based_pipeline.m`

Verify the real Engine and active API with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/check_matlab_engine.ps1
```

Then run one real, read-only dataset image through the complete pipeline with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/test_matlab_pipeline.ps1
```

Results are written to `outputs/matlab_test/`. See
`docs/matlab_pipeline.md`, `docs/system_architecture.md`, and
`docs/report_mapping.md` for the exact flow and response mapping.

## Preserved implementation

The original Flask/MATLAB implementation is preserved under `legacy/`. The older root, `functions/`, and MATLAB scripts also remain untouched as compatibility/reference copies, but the active web service calls only the canonical `matlab/` tree.

The held-out MobileNetV2 result is 96.60% accuracy and 96.53% macro F1; MATLAB rule accuracy is 63.95%. External-data validation remains recommended, and one high-confidence test error shows that confidence is not a correctness guarantee.

The web interface stores local analysis history and separate user feedback in
SQLite. Corrections do not trigger retraining. Local downloadable HTML reports and
the FreshSight Recommendation Assistant are available after analysis. Grad-CAM,
administrator approval, automatic retraining, rollback, and deployment remain
future stages.

## Professional Analysis Dashboard

The main analysis page now presents results in a tabbed dashboard with a precautionary Food Safety Assessment, AI decision explanation, MATLAB visual pipeline, exclusive colour distribution, damage evidence, segmentation reliability, recommendation guidance, feedback controls, and downloadable report access. MobileNetV2 remains the primary classifier; MATLAB remains a supporting measurement subsystem.

Food-safety labels are precautionary visual-screening guidance only:

- Safe to Consume
- Ripen Further
- Unsafe to Consume
- Manual Inspection Required

FreshSight does not directly detect bacteria or certify internal food safety.


## Local Dataset Setup

The FreshSight dataset is not included in this repository.

Create the following local structure:

dataset/
├── fresh/
├── unripe/
└── rotten/

Then run:

powershell -ExecutionPolicy Bypass -File inspect_dataset.ps1

powershell -ExecutionPolicy Bypass -File scripts/run_dataset_split.ps1

## Model Checkpoint

The trained MobileNetV2 checkpoint is not included in the public repository.

Expected local path:

ai/checkpoints/mobilenetv2_cleaned_baseline/best_model.pth