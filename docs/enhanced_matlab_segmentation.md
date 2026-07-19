# Enhanced MATLAB segmentation and reliability gate

## Audit result

Before this phase, `segment_papaya.m` used a broad HSV mask and retained its largest connected component. `extract_papaya_features.m` contained a useful inner erosion, but there was no background-removal function, segmentation-quality schema, HSV/Lab visualization function, exclusive colour classifier, or reliability gate. Consequently, unreliable percentages could reach the rule classifier and recommendation assistant.

The active flow is now:

`run_freshsight_api` → preprocessing → scored component segmentation → quality gate → inner mask → HSV/Lab/colour/texture analysis → filtered damage analysis → reliability-aware MATLAB rules → JSON serialization.

MobileNetV2 remains the primary classifier and is unchanged.

## Component selection

Candidate components are scored using relative area, distance from image centre, solidity, eccentricity, major/minor-axis ratio, border contact, and colour consistency. Thresholds live in `matlab/config/segmentation_config.m`. They are initial development thresholds, not validated accuracy claims.

All percentage calculations use `papayaMaskInner`. Colour priority is white/mold, dark, red lesion, brown, orange, yellow, green, then unclassified. Assigning a pixel removes it from subsequent categories.

## Reliability behavior

`good` segmentation produces normal supporting measurements. `acceptable` segmentation produces measurements and a low-reliability MATLAB class, forcing review. `poor` or `failed` segmentation returns unavailable numeric measurements and an unavailable MATLAB rule class. Python applies the same gate defensively before rendering or recommendation generation.

## Controlled development set

`config/development_segmentation_samples.json` declares ten current manifest entries. Every entry is verified against `dataset_manifest.csv`, and only `train` or `validation` rows are accepted. Scenario labels marked “candidate” must be confirmed visually when reviewing outputs; they are not ground-truth scene annotations.

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/test_enhanced_matlab_segmentation.ps1
```

Results are written under `outputs/matlab_segmentation_test/<sample>/`. Review every raw, clean, and inner mask before claiming an accuracy improvement or changing a threshold. The runner never alters thresholds and rejects held-out test rows.

## Known limitations

This is classical colour/shape segmentation, not semantic instance segmentation. A centred leaf or neighbouring fruit with similar colour and geometry can still win component scoring. Strong shadows, severe occlusion, cropped fruit, unusual cultivars, and low colour contrast can produce acceptable/poor results. Quality gating reduces misleading measurements but cannot guarantee correct isolation. White mold and specular reflection separation remains heuristic, as does stem exclusion.
