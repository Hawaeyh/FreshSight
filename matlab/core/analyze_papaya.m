function result = analyze_papaya(img)
%ANALYZE_PAPAYA Enhanced reliability-gated supporting MATLAB pipeline.
startedAt = tic; cfg = segmentation_config(); processed = preprocess_papaya_image(img);
segmentation = segment_papaya(processed.rgb, cfg);
hsvAnalysis = create_hsv_visualizations(processed.rgb, segmentation.papayaMaskInner);
labAnalysis = create_lab_visualizations(processed.rgb, segmentation.papayaMaskInner);
colourAnalysis = detect_papaya_colours(processed.rgb, segmentation.papayaMaskInner, cfg);
textureAnalysis = analyze_papaya_texture(processed.rgb, segmentation.papayaMaskInner);
features = extract_papaya_features(processed.rgb, segmentation, colourAnalysis, textureAnalysis, cfg);
quality = classify_papaya_freshness(features, segmentation.quality);
highlighted = create_damage_highlight(processed.rgb, features.filteredDamageMask);
isGood = string(segmentation.quality.status) == "good"; isUsable = segmentation.quality.reliable;
reliability.segmentationReliable = isUsable; reliability.colourReliable = isUsable;
reliability.damageReliable = features.damageReliability; reliability.lesionReliable = isUsable;
reliability.moldReliable = isUsable; reliability.textureReliable = isUsable;
reliability.matlabClassReliable = isGood && string(quality.reliability) == "reliable";
reliability.level = char(quality.reliability); reliability.reasons = segmentation.quality.warnings;
result.original = img; result.processed = processed; result.segmentation = segmentation;
result.papayaMask = segmentation.papayaMaskClean; result.hsvAnalysis = hsvAnalysis; result.labAnalysis = labAnalysis;
result.colourAnalysis = colourAnalysis; result.textureAnalysis = textureAnalysis; result.features = features;
result.quality = quality; result.measurementReliability = reliability; result.highlighted = highlighted; result.processingTime = toc(startedAt);
end
