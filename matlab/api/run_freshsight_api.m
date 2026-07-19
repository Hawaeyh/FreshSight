function jsonResult = run_freshsight_api(imagePath)
%RUN_FRESHSIGHT_API JSON entry point for enhanced supporting MATLAB analysis.
if nargin < 1 || strlength(string(imagePath)) == 0, error('FreshSightAPI:MissingImagePath', 'An image path is required.'); end
imagePath = char(imagePath); if ~isfile(imagePath), error('FreshSightAPI:ImageNotFound', 'Image file does not exist: %s', imagePath); end
matlabRoot = fileparts(fileparts(mfilename('fullpath'))); addpath(genpath(matlabRoot));
result = analyze_papaya(imread(imagePath)); f = result.features; s = result.segmentation; c = result.colourAnalysis; t = result.textureAnalysis;
apiResult.status = 'success'; apiResult.error = ''; apiResult.source_image_path = imagePath;
apiResult.rule_class = char(result.quality.status); apiResult.grade = char(result.quality.grade);
apiResult.freshness_score = result.quality.freshnessScore; apiResult.suggestion = char(result.quality.suggestion); apiResult.processing_time_seconds = result.processingTime;
apiResult.rule_reliability = char(result.quality.reliability); apiResult.rule_reliability_score = result.quality.reliabilityScore; apiResult.rule_reasons = result.quality.reasons;
apiResult.segmentation_quality.status = char(s.quality.status); apiResult.segmentation_quality.score = s.quality.score;
apiResult.segmentation_quality.papaya_area_percentage = s.quality.papayaAreaPercentage; apiResult.segmentation_quality.border_touching_percentage = s.quality.borderTouchingPercentage;
apiResult.segmentation_quality.solidity = s.quality.solidity; apiResult.segmentation_quality.eccentricity = s.quality.eccentricity;
apiResult.segmentation_quality.aspect_ratio = s.quality.aspectRatio; apiResult.segmentation_quality.component_count = s.quality.componentCount;
apiResult.segmentation_quality.warnings = s.quality.warnings;
apiResult.measurement_reliability.segmentation_reliable = result.measurementReliability.segmentationReliable;
apiResult.measurement_reliability.colour_reliable = result.measurementReliability.colourReliable;
apiResult.measurement_reliability.damage_reliable = result.measurementReliability.damageReliable;
apiResult.measurement_reliability.lesion_reliable = result.measurementReliability.lesionReliable;
apiResult.measurement_reliability.mold_reliable = result.measurementReliability.moldReliable;
apiResult.measurement_reliability.texture_reliable = result.measurementReliability.textureReliable;
apiResult.measurement_reliability.matlab_class_reliable = result.measurementReliability.matlabClassReliable;
apiResult.measurement_reliability.level = result.measurementReliability.level; apiResult.measurement_reliability.reasons = result.measurementReliability.reasons;
apiResult.bounding_box = s.boundingBox; apiResult.centroid = s.centroid;
apiResult.colour_analysis.green_percentage = f.greenPercentage; apiResult.colour_analysis.yellow_percentage = f.yellowPercentage;
apiResult.colour_analysis.orange_percentage = f.orangePercentage; apiResult.colour_analysis.brown_percentage = f.brownPercentage;
apiResult.colour_analysis.dark_percentage = f.darkPercentage; apiResult.colour_analysis.white_percentage = f.whiteMoldPercentage;
apiResult.colour_analysis.red_lesion_percentage = f.lesionPercentage; apiResult.colour_analysis.unclassified_percentage = f.unclassifiedPercentage;
apiResult.colour_analysis.dominant_colour = char(c.dominantColour); apiResult.colour_analysis.secondary_colour = char(c.secondaryColour);
apiResult.texture_analysis.mean_entropy = t.meanEntropy; apiResult.texture_analysis.edge_density = t.edgeDensity;
apiResult.texture_analysis.glcm_contrast = t.glcmContrast; apiResult.texture_analysis.glcm_correlation = t.glcmCorrelation;
apiResult.texture_analysis.glcm_energy = t.glcmEnergy; apiResult.texture_analysis.glcm_homogeneity = t.glcmHomogeneity; apiResult.texture_analysis.rough_percentage = f.roughPercentage;
apiResult.lab_analysis.mean_l = result.labAnalysis.mean.L; apiResult.lab_analysis.mean_a = result.labAnalysis.mean.a; apiResult.lab_analysis.mean_b = result.labAnalysis.mean.b;
apiResult.lab_analysis.std_l = result.labAnalysis.standardDeviation.L; apiResult.lab_analysis.std_a = result.labAnalysis.standardDeviation.a; apiResult.lab_analysis.std_b = result.labAnalysis.standardDeviation.b;
apiResult.measurements.green_percentage = f.greenPercentage; apiResult.measurements.yellow_percentage = f.yellowPercentage; apiResult.measurements.orange_percentage = f.orangePercentage;
apiResult.measurements.brown_percentage = f.brownPercentage; apiResult.measurements.dark_percentage = f.darkPercentage; apiResult.measurements.white_mold_percentage = f.whiteMoldPercentage;
apiResult.measurements.lesion_percentage = f.lesionPercentage; apiResult.measurements.rough_percentage = f.roughPercentage; apiResult.measurements.damage_percentage = f.damagePercentage;
apiResult.measurements.healthy_percentage = f.healthyPercentage; apiResult.measurements.largest_damage_percentage = f.largestDamagePercentage;
apiResult.measurements.largest_lesion_percentage = f.largestLesionPercentage;
apiResult.measurements.brown_region_count = f.brownRegionCount; apiResult.measurements.dark_region_count = f.darkRegionCount;
apiResult.measurements.white_mold_region_count = f.whiteMoldRegionCount; apiResult.measurements.damage_region_count = f.damageRegionCount; apiResult.measurements.lesion_region_count = f.lesionRegionCount;
apiResult.damage_percentage = f.damagePercentage; apiResult.healthy_percentage = f.healthyPercentage; apiResult.damage_severity = char(f.damageSeverity); apiResult.damage_reliability = f.damageReliability;
apiResult.rule_scores.unripe = result.quality.scores.unripe; apiResult.rule_scores.fresh = result.quality.scores.fresh; apiResult.rule_scores.rotten = result.quality.scores.rotten;
apiResult.rule_evidence = result.quality.ruleEvidence;
apiResult.damage_evidence.brown_decay_percentage = f.damageEvidence.brownDecayPercentage;
apiResult.damage_evidence.dark_decay_percentage = f.damageEvidence.darkDecayPercentage;
apiResult.damage_evidence.mold_percentage = f.damageEvidence.moldPercentage;
apiResult.damage_evidence.lesion_percentage = f.damageEvidence.lesionPercentage;
apiResult.damage_evidence.abnormal_texture_percentage = f.damageEvidence.abnormalTexturePercentage;
apiResult.damage_evidence.excluded_stem_percentage = f.damageEvidence.excludedStemPercentage;
apiResult.damage_evidence.excluded_shadow_percentage = f.damageEvidence.excludedShadowPercentage;
apiResult.damage_evidence.excluded_reflection_percentage = f.damageEvidence.excludedReflectionPercentage;
apiResult.damage_evidence.stem_detected = f.damageEvidence.stemDetected;
apiResult.damage_evidence.stem_confidence = f.damageEvidence.stemConfidence;
apiResult.damage_evidence.shadow_confidence = f.damageEvidence.shadowConfidence;
apiResult.images.original = image_to_base64(result.processed.rgb); apiResult.images.background_removed = image_to_base64(s.backgroundRemoved);
apiResult.images.transparent_background = image_to_base64_alpha(result.processed.rgb, s.papayaMaskClean);
apiResult.images.papaya_mask = image_to_base64(s.papayaMaskClean); apiResult.images.papaya_mask_raw = image_to_base64(s.papayaMaskRaw);
apiResult.images.papaya_mask_clean = image_to_base64(s.papayaMaskClean); apiResult.images.papaya_mask_inner = image_to_base64(s.papayaMaskInner);
apiResult.images.hue_channel = image_to_base64(result.hsvAnalysis.hueChannel); apiResult.images.saturation_channel = image_to_base64(result.hsvAnalysis.saturationChannel);
apiResult.images.value_channel = image_to_base64(result.hsvAnalysis.valueChannel); apiResult.images.hsv_visualization = image_to_base64(result.hsvAnalysis.hsvVisualization);
apiResult.images.l_channel = image_to_base64(result.labAnalysis.LChannel); apiResult.images.a_channel = image_to_base64(result.labAnalysis.aChannel); apiResult.images.b_channel = image_to_base64(result.labAnalysis.bChannel);
apiResult.images.green_mask = image_to_base64(c.masks.green); apiResult.images.yellow_mask = image_to_base64(c.masks.yellow); apiResult.images.orange_mask = image_to_base64(c.masks.orange);
apiResult.images.brown_mask = image_to_base64(c.masks.brown); apiResult.images.dark_mask = image_to_base64(c.masks.dark);
apiResult.images.white_colour_mask = image_to_base64(c.masks.whiteMold); apiResult.images.red_colour_mask = image_to_base64(c.masks.redLesion);
apiResult.images.white_mold_mask = image_to_base64(f.whiteMoldMask); apiResult.images.red_lesion_mask = image_to_base64(f.lesionMask);
apiResult.images.unclassified_mask = image_to_base64(c.masks.unclassified); apiResult.images.colour_segmentation_overlay = image_to_base64(c.combinedOverlay);
apiResult.images.texture_map = image_to_base64(t.textureMap); apiResult.images.edge_map = image_to_base64(t.edgeMap); apiResult.images.rough_area_mask = image_to_base64(t.roughAreaMask);
apiResult.images.raw_damage_mask = image_to_base64(f.rawDamageMask); apiResult.images.filtered_damage_mask = image_to_base64(f.filteredDamageMask);
apiResult.images.damage_mask = image_to_base64(f.filteredDamageMask); apiResult.images.damage_highlight = image_to_base64(result.highlighted);
apiResult.images.stem_mask = image_to_base64(f.stemMask); apiResult.images.shadow_mask = image_to_base64(f.shadowMask);
apiResult.images.reflection_mask = image_to_base64(f.reflectionMask); apiResult.images.brown_decay_mask = image_to_base64(f.brownDecayMask);
apiResult.images.dark_decay_mask = image_to_base64(f.darkDecayMask); apiResult.images.lesion_mask = image_to_base64(f.lesionMask);
apiResult.images.abnormal_texture_mask = image_to_base64(f.abnormalTextureMask);
apiResult.images.combined_damage_evidence = image_to_base64(f.combinedDamageEvidence);
% Compatibility aliases.
apiResult.images.green_area = apiResult.images.green_mask; apiResult.images.yellow_area = apiResult.images.yellow_mask; apiResult.images.brown_area = apiResult.images.brown_mask;
apiResult.images.white_mold = apiResult.images.white_mold_mask; apiResult.images.dark_area = apiResult.images.dark_mask; apiResult.images.lesion_area = apiResult.images.red_lesion_mask; apiResult.images.rough_area = apiResult.images.rough_area_mask;
jsonResult = jsonencode(apiResult, 'ConvertInfAndNaN', true);
end

function base64String = image_to_base64(imgMatrix)
if islogical(imgMatrix), imgMatrix = uint8(imgMatrix) * 255; end
tempFile = [tempname, '.png']; cleanup = onCleanup(@() delete_if_present(tempFile)); imwrite(imgMatrix, tempFile); base64String = read_base64(tempFile); clear cleanup
end

function base64String = image_to_base64_alpha(rgb, mask)
tempFile = [tempname, '.png']; cleanup = onCleanup(@() delete_if_present(tempFile)); imwrite(rgb, tempFile, 'Alpha', uint8(mask) * 255); base64String = read_base64(tempFile); clear cleanup
end

function value = read_base64(path)
fid = fopen(path, 'r'); if fid == -1, error('FreshSightAPI:FileReadError', 'Unable to read temporary PNG file.'); end
cleanup = onCleanup(@() fclose(fid)); bytes = fread(fid, Inf, '*uint8'); value = char(matlab.net.base64encode(bytes)); clear cleanup
end

function delete_if_present(path)
if isfile(path), delete(path); end
end
