function colour = detect_papaya_colours(img, papayaMask, cfg)
%DETECT_PAPAYA_COLOURS Exclusive colour classes inside the inner fruit mask.
% Priority: white/mold, dark, red lesion, brown, orange, yellow, green, unclassified.
if nargin < 3, cfg = segmentation_config(); end
hsvImg = rgb2hsv(img); labImg = rgb2lab(img); gray = im2double(rgb2gray(img));
H = hsvImg(:, :, 1); S = hsvImg(:, :, 2); V = hsvImg(:, :, 3);
a = labImg(:, :, 2); b = labImg(:, :, 3); texture = stdfilt(gray, true(5));
validArea = nnz(papayaMask); minimumPixels = max(8, round(validArea * cfg.minimumColourRegionFraction));
available = papayaMask;
specular = papayaMask & S < 0.10 & V > 0.92 & texture < 0.018;
candidates.whiteMold = papayaMask & S < 0.20 & V > 0.68 & texture > 0.018 & ~specular;
candidates.dark = papayaMask & V < 0.25 & texture > 0.010;
candidates.redLesion = papayaMask & (H >= 0.94 | H < 0.025) & S > 0.32 & a > 10 & texture > 0.012;
candidates.brown = papayaMask & H >= 0.015 & H < 0.105 & S > 0.28 & V >= 0.12 & V < 0.58 & b > 5;
candidates.orange = papayaMask & H >= 0.025 & H < 0.105 & S > 0.20 & V >= 0.48;
candidates.yellow = papayaMask & H >= 0.105 & H < 0.19 & S > 0.16 & V >= 0.35;
candidates.green = papayaMask & H >= 0.19 & H <= 0.49 & S > 0.14 & V >= 0.16;
priority = {'whiteMold', 'dark', 'redLesion', 'brown', 'orange', 'yellow', 'green'};
for index = 1:numel(priority)
    name = priority{index}; mask = candidates.(name) & available;
    mask = bwareaopen(mask, minimumPixels); colour.masks.(name) = mask; available(mask) = false;
end
colour.masks.unclassified = available;
names = [priority, {'unclassified'}]; percentages = zeros(1, numel(names));
for index = 1:numel(names)
    name = names{index}; mask = colour.masks.(name);
    percentages(index) = 100 * nnz(mask) / max(validArea, 1);
    cc = bwconncomp(mask); colour.regionCount.(name) = cc.NumObjects;
    componentStats = regionprops(cc, 'Area');
    if isempty(componentStats), colour.largestRegionPercentage.(name) = 0;
    else, colour.largestRegionPercentage.(name) = 100 * max([componentStats.Area]) / max(validArea, 1); end
    colour.percentage.(name) = min(100, max(0, percentages(index)));
end
[~, order] = sort(percentages(1:7), 'descend'); displayNames = {'White/mold', 'Dark', 'Red lesion', 'Brown', 'Orange', 'Yellow', 'Green'};
if validArea == 0, colour.dominantColour = "Unavailable"; colour.secondaryColour = "Unavailable";
else, colour.dominantColour = displayNames{order(1)}; colour.secondaryColour = displayNames{order(2)}; end
palette = uint8([245 245 245; 25 25 25; 220 30 45; 120 65 25; 245 130 25; 250 220 35; 45 175 70; 140 140 140]);
overlay = im2uint8(img); blend = 0.55;
for index = 1:numel(names)
    mask = colour.masks.(names{index});
    for channel = 1:3
        plane = overlay(:, :, channel); plane(mask) = uint8((1-blend) * double(plane(mask)) + blend * double(palette(index, channel))); overlay(:, :, channel) = plane;
    end
end
colour.combinedOverlay = overlay; colour.priorityOrder = displayNames; colour.totalPercentage = sum(percentages); colour.validPixelCount = validArea;
end
