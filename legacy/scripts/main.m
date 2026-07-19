clc;
clear;
close all;

projectRoot = fileparts(fileparts(mfilename("fullpath")));

addpath(fullfile(projectRoot, "functions"));

[file, path] = uigetfile( ...
    {'*.jpg;*.jpeg;*.png', 'Image Files (*.jpg,*.jpeg,*.png)'}, ...
    'Select Papaya Image');

if isequal(file,0)
    disp("No image selected.");
    return;
end

imagePath = fullfile(path, file);
img = imread(imagePath);

result = analyzePapayaNew(img);

figure("Name", "FreshSight V2 Detection", "NumberTitle", "off");

subplot(2,4,1);
imshow(result.processed.original);
title("Original");

subplot(2,4,2);
imshow(result.papayaMask);
title("Papaya Mask");

subplot(2,4,3);
imshow(result.features.greenMask);
title("Green Area");

subplot(2,4,4);
imshow(result.features.yellowMask);
title("Yellow Area");

subplot(2,4,5);
imshow(result.features.brownMask);
title("Brown Area");

subplot(2,4,6);
imshow(result.features.whiteMask);
title("White Mold");

subplot(2,4,7);
imshow(result.highlighted);
title("Damage Highlight");

subplot(2,4,8);
axis off;

resultText = sprintf( ...
['FreshSight Result\n\n' ...
 'Status: %s\n' ...
 'Grade: %s\n\n' ...
 'Freshness Score: %.2f%%\n' ...
 'Damage Area: %.2f%%\n' ...
 'Healthy Area: %.2f%%\n\n' ...
 'Green Area: %.2f%%\n' ...
 'Yellow Area: %.2f%%\n' ...
 'Brown Area: %.2f%%\n' ...
 'Dark Area: %.2f%%\n' ...
 'White Mold: %.2f%%\n\n' ...
 'Processing Time: %.2f sec\n\n' ...
 'Suggestion:\n%s'], ...
 char(result.quality.status), ...
 char(result.quality.grade), ...
 result.quality.freshnessScore, ...
 result.features.damagePercentage, ...
 result.features.healthyPercentage, ...
 result.features.greenPercentage, ...
 result.features.yellowPercentage, ...
 result.features.brownPercentage, ...
 result.features.darkPercentage, ...
 result.features.whiteMoldPercentage, ...
 result.processingTime, ...
 char(result.quality.suggestion));

text(0.02, 0.95, resultText, ...
    "FontSize", 10, ...
    "FontWeight", "bold", ...
    "VerticalAlignment", "top");

fprintf("\n========== FreshSight V2 Result ==========\n");
fprintf("Image            : %s\n", file);
fprintf("Status           : %s\n", char(result.quality.status));
fprintf("Grade            : %s\n", char(result.quality.grade));
fprintf("Freshness Score  : %.2f%%\n", result.quality.freshnessScore);
fprintf("Damage Area      : %.2f%%\n", result.features.damagePercentage);
fprintf("Healthy Area     : %.2f%%\n", result.features.healthyPercentage);
fprintf("Green Area       : %.2f%%\n", result.features.greenPercentage);
fprintf("Yellow Area      : %.2f%%\n", result.features.yellowPercentage);
fprintf("Brown Area       : %.2f%%\n", result.features.brownPercentage);
fprintf("Dark Area        : %.2f%%\n", result.features.darkPercentage);
fprintf("White Mold       : %.2f%%\n", result.features.whiteMoldPercentage);
fprintf("Processing Time  : %.2f seconds\n", result.processingTime);
fprintf("Suggestion       : %s\n", char(result.quality.suggestion));
fprintf("==========================================\n");