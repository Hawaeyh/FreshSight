# MATLAB rule-based pipeline

The active MATLAB subsystem is rooted at `matlab/`. Add the whole tree with
`addpath(genpath(<project>/matlab))`; do not add the root `functions/` directory
for new integrations.

## Active call flow

`run_freshsight_api` → `analyze_papaya` → `preprocess_papaya_image` →
`segment_papaya` → `extract_papaya_features` →
`classify_papaya_freshness` → `create_damage_highlight`

| File | Declared function | Inputs | Outputs | Direct calls | Purpose |
|---|---|---|---|---|---|
| `matlab/api/run_freshsight_api.m` | `run_freshsight_api` | image path | JSON text | `analyze_papaya` | Validates/reads an image and serializes real results and PNG artifacts. |
| `matlab/core/analyze_papaya.m` | `analyze_papaya` | image matrix | result struct | all five stages below | Orchestrates and times one analysis. |
| `matlab/preprocessing/preprocess_papaya_image.m` | `preprocess_papaya_image` | image matrix | processed struct | MATLAB image functions | Resize, mild filter, illumination correction, RGB/gray/HSV views. |
| `matlab/segmentation/segment_papaya.m` | `segment_papaya` | RGB image | logical fruit mask | MATLAB image functions | Colour candidate segmentation and largest-region cleanup. |
| `matlab/features/extract_papaya_features.m` | `extract_papaya_features` | RGB image, fruit mask | features struct | MATLAB image functions | Actual colour, damage, lesion, texture, region metrics and masks. |
| `matlab/classification/classify_papaya_freshness.m` | `classify_papaya_freshness` | features struct | quality struct | none | Existing deterministic Fresh/Unripe/Rotten rules and scores. |
| `matlab/visualization/create_damage_highlight.m` | `create_damage_highlight` | RGB image, damage mask | highlighted RGB image | none | Marks detected damage red. |
| `matlab/scripts/run_single_image_demo.m` | script | selected image | figure/console | `analyze_papaya` | Interactive local demonstration; requires MATLAB desktop for the file picker/figure. |
| `matlab/scripts/evaluate_rule_based_pipeline.m` | script | class folders | evaluation CSV | `analyze_papaya` | Offline rule evaluation; not invoked by Flask. |

## Requirements and compatibility

- MATLAB R2025b and MATLAB Engine for Python 25.2.2.
- Image Processing Toolbox is required by resizing, filtering, colour conversion,
  adaptive histogram equalization, morphology, connected components, texture, and
  region-statistics operations.
- The API uses base MATLAB JSON support and `matlab.net.base64encode`.
- The old root, `functions/`, and `scripts/*.m` implementations remain unchanged,
  with identical preserved copies under `legacy/`. They are compatibility/reference
  code only. Their camel-case names are not called by the active web service.
- The old `scripts/main.m` is interactive. The old `scripts/test.m` writes to
  `results/`, embeds lowercase class names, and calls the old API names. It is not
  part of the active pipeline.

Run the real integration checks only from the project root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/check_matlab_engine.ps1
powershell -ExecutionPolicy Bypass -File scripts/test_matlab_pipeline.ps1
```

The second command reads one real supported dataset image and writes decoded copies
of returned MATLAB images plus `matlab_test_result.json` under
`outputs/matlab_test/`. It never modifies the source image.

## Held-out rule-based evaluation

After the one-image integration check, run the complete rule evaluation with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_matlab_evaluation.ps1
```

The runner validates and reads only `test` rows from
`evaluation/outputs/dataset_manifest.csv` (42 Fresh, 53 Unripe, 52 Rotten), then
reuses one Engine session for all 147 calls to `evaluate_rule_based_pipeline`.
That MATLAB helper calls `analyze_papaya`, so every canonical processing stage,
including damage-highlight creation, still executes; Base64 images are omitted from
the evaluation response for efficiency. Source images are read-only.

Metrics use successful records only. Failures retain blank measurement/prediction
fields and their real error messages, and are written separately rather than being
converted to an invented class or zero-valued measurements. Outputs are stored in
`evaluation/outputs/matlab_rule_based/`.

Report-ready artifacts are organized into `figures/`, `tables/`, `reports/`, and
`previews/`. They include 300-DPI performance and distribution plots, per-class
feature statistics, all off-diagonal class-pair counts, a reconstructed activation
summary of the unchanged classifier rules, and a contact sheet containing up to the
first 25 misclassified source images. The generated Markdown report summarizes the
observations but deliberately does not recommend or apply threshold changes.

The earlier root-level CSV, JSON, and `matlab_*confusion_matrix.png` names remain in
place for compatibility with the prepared evaluation workflow.
