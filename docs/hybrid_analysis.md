# Hybrid analysis

Every upload runs two independent subsystems. MobileNetV2 produces the primary
Fresh/Unripe/Rotten classification and probabilities. MATLAB produces a separate
rule class, segmentation masks, colour and texture measurements, lesion/mold
indicators, damage percentage, healthy percentage, and highlighted regions.

Manual review is required when AI or MATLAB is unavailable, AI confidence is below
0.65, the top-two probability margin is below 0.10, or the subsystem classes
disagree. These warnings never change the MobileNetV2 prediction. Missing values
are not converted to zero.

The FreshSight Recommendation Assistant combines the visible results into
transparent handling guidance. It is not food-safety certification.
