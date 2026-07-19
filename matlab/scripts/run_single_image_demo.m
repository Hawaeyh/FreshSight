%RUN_SINGLE_IMAGE_DEMO Interactively run the canonical rule-based pipeline.
clc;
clear;
close all;

matlabRoot = fileparts(fileparts(mfilename("fullpath")));
addpath(genpath(matlabRoot));
[file, folder] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff', 'Supported images'}, ...
    'Select Papaya Image');
if isequal(file, 0)
    disp("No image selected.");
    return;
end

result = analyze_papaya(imread(fullfile(folder, file)));
fprintf("Rule class       : %s\n", char(result.quality.status));
fprintf("Grade            : %s\n", char(result.quality.grade));
fprintf("Freshness score  : %.2f%%\n", result.quality.freshnessScore);
fprintf("Damage area      : %.2f%%\n", result.features.damagePercentage);
fprintf("Healthy area     : %.2f%%\n", result.features.healthyPercentage);
fprintf("Processing time  : %.3f seconds\n", result.processingTime);

figure("Name", "FreshSight MATLAB Rule-Based Analysis", "NumberTitle", "off");
subplot(2, 2, 1); imshow(result.processed.original); title("Original");
subplot(2, 2, 2); imshow(result.papayaMask); title("Papaya Mask");
subplot(2, 2, 3); imshow(result.features.damageMask); title("Damage Mask");
subplot(2, 2, 4); imshow(result.highlighted); title("Damage Highlight");
