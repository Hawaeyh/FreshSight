function cfg = damage_config()
%DAMAGE_CONFIG Development-only evidence thresholds (train/validation notes).
cfg.minimumEvidenceRegionFraction = 0.0008;
cfg.minimumMoldRegionFraction = 0.0012;
cfg.minimumLesionRegionFraction = 0.0008;
cfg.minimumTextureRegionFraction = 0.0010;
cfg.stemMaximumAreaPercentage = 5.0;
cfg.stemEndpointBandFraction = 0.22;
cfg.shadowValueRatio = 0.72;
cfg.shadowLightnessRatio = 0.75;
cfg.shadowMaximumTexture = 0.035;
cfg.shadowMaximumLocalContrast = 0.09;
cfg.reflectionMaximumSaturation = 0.16;
cfg.reflectionMinimumValue = 0.84;
cfg.reflectionMaximumTexture = 0.020;
cfg.darkDecayMaximumValue = 0.30;
cfg.brownDecayMaximumValue = 0.62;
cfg.minimumDecayTexture = 0.025;
cfg.minimumDecayLocalContrast = 0.07;
cfg.severityMinimalMaximum = 3.0;
cfg.severityMildMaximum = 10.0;
cfg.severityModerateMaximum = 25.0;
cfg.severeLargestRegionMinimum = 8.0;
end
