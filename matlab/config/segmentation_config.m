function cfg = segmentation_config()
%SEGMENTATION_CONFIG Documented development thresholds for papaya isolation.
% Tune only with training/validation development images, never held-out test data.
cfg.minimumCandidateAreaFraction = 0.01;
cfg.minimumAcceptedAreaPercentage = 8;
cfg.maximumAcceptedAreaPercentage = 85;
cfg.maximumBorderTouchPercentage = 18;
cfg.minimumSolidity = 0.68;
cfg.minimumGoodScore = 0.72;
cfg.minimumAcceptableScore = 0.50;
cfg.similarComponentAreaRatio = 0.70;
cfg.maximumSimilarComponents = 2;
cfg.minimumComponentPixels = 300;
cfg.openRadius = 3;
cfg.closeRadius = 4;
cfg.smoothRadius = 4;
cfg.innerErosionRadius = 3;
cfg.neutralBackgroundValue = 255;
cfg.minimumDamageRegionFraction = 0.0008;
cfg.minimumColourRegionFraction = 0.0002;

% Component-selection weights. These affect ranking only; the reliability
% gate and its acceptance thresholds remain unchanged.
cfg.weight_area = 0.15;
cfg.weight_solidity = 0.16;
cfg.weight_eccentricity = 0.08;
cfg.weight_extent = 0.10;
cfg.weight_border_contact = 0.18;
cfg.weight_center_distance = 0.13;
cfg.weight_aspect_ratio = 0.10;
cfg.weight_colour_consistency = 0.10;

cfg.activeContourIterations = 80;
cfg.activeContourSmoothFactor = 1.5;
cfg.leafMinimumWidthFraction = 0.012;
cfg.leafGreenVarianceThreshold = 0.035;
cfg.centralSeedWidthFraction = 0.46;
cfg.centralSeedHeightFraction = 0.64;
cfg.componentSelectionMinimumSolidity = 0.90;
cfg.maximumConvexCompletionRatio = 1.50;
end
