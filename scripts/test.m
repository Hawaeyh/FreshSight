clc;
clear;
close all;

%% Project setup
projectRoot = fileparts(fileparts(mfilename("fullpath")));
addpath(fullfile(projectRoot, "functions"));

datasetRoot = fullfile(projectRoot, "dataset");
resultsFolder = fullfile(projectRoot, "results");

if ~exist(resultsFolder, "dir")
    mkdir(resultsFolder);
end

% Removed "semi_fresh" to use only 3 classes
classes = ["unripe", "fresh", "rotten"];

%% Create empty results table
T = table( ...
    'Size', [0 19], ...
    'VariableTypes', { ...
        'string','string','string','logical', ...
        'double','double','double','double','double','double', ...
        'double','double','double','double','double','double', ...
        'double','double','double'}, ...
    'VariableNames', { ...
        'filename', ...
        'correctLabel', ...
        'predictedLabel', ...
        'isCorrect', ...
        'freshnessScore', ...
        'damagePercentage', ...
        'healthyPercentage', ...
        'greenPercentage', ...
        'yellowPercentage', ...
        'brownPercentage', ...
        'darkPercentage', ...
        'whiteMoldPercentage', ...
        'greenPixels', ...
        'yellowPixels', ...
        'brownPixels', ...
        'unripeScore', ...
        'freshScore', ...
        'rottenScore', ...
        'processingTime'});

%% Process each class folder
for c = 1:length(classes)
    correctLabel = classes(c);
    classFolder = fullfile(datasetRoot, correctLabel);
    
    if ~exist(classFolder, "dir")
        warning("Folder %s does not exist. Skipping...", classFolder);
        continue;
    end

    imageFiles = [
        dir(fullfile(classFolder, "*.jpg"));
        dir(fullfile(classFolder, "*.jpeg"));
        dir(fullfile(classFolder, "*.png"));
        dir(fullfile(classFolder, "*.JPEG"));
        dir(fullfile(classFolder, "*.JPG"))
    ];

    for i = 1:length(imageFiles)
        file = imageFiles(i).name;
        imagePath = fullfile(classFolder, file);
        
        try
            img = imread(imagePath);
            result = analyzePapayaNew(img);
            
            predictedLabel = lower(string(result.quality.status));
            isCorrect = (correctLabel == predictedLabel);
            
            greenPixels = sum(result.features.greenMask(:));
            yellowPixels = sum(result.features.yellowMask(:));
            brownPixels = sum(result.features.brownMask(:));
            
            newRow = { ...
                string(file), ...
                correctLabel, ...
                predictedLabel, ...
                isCorrect, ...
                result.quality.freshnessScore, ...
                result.features.damagePercentage, ...
                result.features.healthyPercentage, ...
                result.features.greenPercentage, ...
                result.features.yellowPercentage, ...
                result.features.brownPercentage, ...
                result.features.darkPercentage, ...
                result.features.whiteMoldPercentage, ...
                greenPixels, ...
                yellowPixels, ...
                brownPixels, ...
                result.quality.scores.unripe, ...
                result.quality.scores.fresh, ...
                result.quality.scores.rotten, ...
                result.processingTime};
                
            T = [T; newRow];
            
            fprintf("%s | Correct: %s | Predicted: %s | Damage: %.2f%% | Match: %d\n", ...
                file, correctLabel, predictedLabel, result.features.damagePercentage, isCorrect);
                
        catch ME
            warning("Error processing %s: %s", file, ME.message);
        end
    end
end

if height(T) == 0
    error("No images found. Check your dataset folders.");
end

accuracy = mean(T.isCorrect) * 100;

fprintf("\n========== FreshSight Dataset Test ==========\n");
fprintf("Total Images     : %d\n", height(T));
fprintf("Correct Results  : %d\n", sum(T.isCorrect));
fprintf("Wrong Results    : %d\n", sum(~T.isCorrect));
fprintf("Overall Accuracy : %.2f%%\n", accuracy);

%% Per-class accuracy
fprintf("\n========== Per-Class Accuracy ===============\n");
for c = 1:length(classes)
    className = classes(c);
    classRows = T.correctLabel == className;
    if any(classRows)
        classAccuracy = mean(T.isCorrect(classRows)) * 100;
        classTotal = sum(classRows);
        classCorrect = sum(T.isCorrect(classRows));
        fprintf("%-12s : %.2f%% (%d/%d)\n", className, classAccuracy, classCorrect, classTotal);
    end
end
fprintf("=============================================\n");

%% Save full result table
outputFile = fullfile(resultsFolder, "dataset_test_results_full.csv");
writetable(T, outputFile);
fprintf("\nResults saved to:\n%s\n", outputFile);

%% Save incorrectly classified images only
wrongResults = T(~T.isCorrect, :);
if height(wrongResults) > 0
    wrongOutputFile = fullfile(resultsFolder, "misclassified_results.csv");
    writetable(wrongResults, wrongOutputFile);
    fprintf("Misclassified results saved to:\n%s\n", wrongOutputFile);
end

%% Bar Chart Visualization (Easy to Understand)
fig = figure('Name', 'FreshSight Classification Performance', 'NumberTitle', 'off', 'Color', 'w');
fig.Position = [150, 150, 800, 500];

% 1. Calculate correct and wrong counts for each class
numClasses = length(classes);
correctCounts = zeros(1, numClasses);
wrongCounts = zeros(1, numClasses);

for c = 1:numClasses
    classRows = T.correctLabel == classes(c);
    correctCounts(c) = sum(T.isCorrect(classRows));
    wrongCounts(c) = sum(~T.isCorrect(classRows));
end

% 2. Draw Grouped Bar Chart
b = bar([correctCounts; wrongCounts]', 'grouped', 'EdgeColor', 'none');

% 3. Color the bars (Green for Correct, Red for Wrong)
b(1).FaceColor = [0.15 0.68 0.38]; % Emerald Green
b(2).FaceColor = [0.90 0.30 0.30]; % Soft Red

% 4. Beautify labels, titles, and axes (Forcing high contrast)
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', ...
    'XTickLabel', {'Unripe', 'Fresh', 'Rotten'}, 'FontSize', 12, 'FontName', 'Arial');

title('Correct vs Wrong Predictions by Class', 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k');
ylabel('Number of Images', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');

% Force legend text and background to be visible
lgd = legend('Correct (Accurate)', 'Wrong (Misclassified)', 'Location', 'northeast', 'FontSize', 11);
set(lgd, 'TextColor', 'k', 'Color', 'w');
grid on;

% 5. Add exact numbers on top of each bar for clarity
for i = 1:2
    xtips = b(i).XEndPoints;
    ytips = b(i).YEndPoints;
    labels = string(b(i).YData);
    text(xtips, ytips, labels, 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
end