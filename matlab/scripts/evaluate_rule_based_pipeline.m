function jsonResult = evaluate_rule_based_pipeline(imagePath)
%EVALUATE_RULE_BASED_PIPELINE Evaluate one image without serializing display images.
% The Python manifest runner calls this function repeatedly in one Engine session.
% analyze_papaya still runs preprocessing, segmentation, feature extraction,
% classification, and damage-highlight creation.

if nargin < 1 || strlength(string(imagePath)) == 0
    error("FreshSightEvaluation:MissingImagePath", "An image path is required.");
end

imagePath = char(imagePath);
if ~isfile(imagePath)
    error("FreshSightEvaluation:ImageNotFound", ...
        "Image file does not exist: %s", imagePath);
end

matlabRoot = fileparts(fileparts(mfilename("fullpath")));
addpath(genpath(matlabRoot));
result = analyze_papaya(imread(imagePath));

evaluation.status = "success";
evaluation.error = "";
evaluation.source_path = imagePath;
evaluation.predicted_class = char(result.quality.status);
evaluation.grade = char(result.quality.grade);
evaluation.freshness_score = result.quality.freshnessScore;
evaluation.green_percentage = result.features.greenPercentage;
evaluation.yellow_percentage = result.features.yellowPercentage;
evaluation.brown_percentage = result.features.brownPercentage;
evaluation.dark_percentage = result.features.darkPercentage;
evaluation.white_mold_percentage = result.features.whiteMoldPercentage;
evaluation.lesion_percentage = result.features.lesionPercentage;
evaluation.rough_percentage = result.features.roughPercentage;
evaluation.damage_percentage = result.features.damagePercentage;
evaluation.healthy_percentage = result.features.healthyPercentage;
evaluation.largest_damage_percentage = result.features.largestDamagePercentage;
evaluation.largest_lesion_percentage = result.features.largestLesionPercentage;
evaluation.unripe_rule_score = result.quality.scores.unripe;
evaluation.fresh_rule_score = result.quality.scores.fresh;
evaluation.rotten_rule_score = result.quality.scores.rotten;
evaluation.processing_time_seconds = result.processingTime;

jsonResult = jsonencode(evaluation);
end
