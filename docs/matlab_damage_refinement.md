# Controlled MATLAB damage-evidence refinement

## Audit

The previous damage mask directly united exclusive brown, dark, white/mold, and red-lesion colour masks. It had no explicit stem, smooth-shadow, or specular-reflection exclusion. Consequently, normal stems, dark-green skin, illumination gradients, and smooth highlights could contribute to damage and supporting Rotten rules.

The segmentation implementation and reliability thresholds are frozen. Damage analysis consumes the existing `papayaMaskInner` and cannot expand beyond it.

## Calibration samples

`config/development_damage_samples.json` declares 24 manually reviewed manifest rows: eight Fresh, eight Unripe, and eight Rotten. All are `train` entries in the current manifest. Notes cover stems, leafy scenes, shadows, reflections, cosmetic blemishes, lesions, mold, severe rot, and different lighting. These are calibration notes—not test metrics or food-safety labels.

## Evidence flow

1. Detect conservative endpoint stem/scar candidates.
2. Detect smooth low-light shadow regions using HSV/Lab, texture, and local contrast.
3. Detect smooth bright low-saturation reflections.
4. Require texture, contrast, lesion, brown, dark, or mold support for decay evidence.
5. Remove small evidence regions.
6. Exclude stem, shadow, reflection, background, and fruit boundary.
7. Combine retained brown decay, dark decay, mold, lesion, and abnormal texture.
8. Calculate severity using total damage and largest connected evidence, preventing many tiny pixels alone from producing Severe.

Thresholds are stored separately in `matlab/config/damage_config.m`. Segmentation weights and reliability thresholds are not read or modified by this stage.

## Supporting rules

Hard Rotten gates require one of:

- clustered mold plus connected damage;
- at least 25% total damage plus an 8% largest connected region;
- dark decay plus connected brown, lesion, or abnormal-texture support;
- connected lesion evidence.

All remaining decisions use combined colour/evidence scores, followed by a clean ripe-colour rescue when connected decay evidence is low. MATLAB remains supporting evidence. A reliable disagreement with Fresh/Unripe AI produces cautious retake and manual-review guidance.

## Controlled test

```powershell
powershell -ExecutionPolicy Bypass -File scripts/test_enhanced_matlab_damage.ps1
```

The runner rejects held-out test rows, never changes thresholds, and writes `outputs/matlab_damage_test/`. It reads the existing ten segmentation results as the before baseline and writes `damage_refinement_comparison.csv` without overwriting them.

Do not claim improved damage accuracy until all evidence masks are visually inspected.

## Limitations

Stem orientation is not guaranteed, shadows can contain texture, severe rot can resemble healthy brown/orange skin, and glare can coexist with mold. These classical heuristics cannot establish food safety. Manual calibration notes are subjective and require review.
