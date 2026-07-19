function features = extractPapayaFeatures(img, papayaMask)
%EXTRACTPAPAYAFEATURES Extract colour, damage, lesion and texture features.

%% Ensure RGB image
if size(img, 3) == 1
    img = cat(3, img, img, img);
end

%% Convert colour spaces
hsvImg = rgb2hsv(img);
grayImg = rgb2gray(img);

H = hsvImg(:, :, 1);
S = hsvImg(:, :, 2);
V = hsvImg(:, :, 3);

grayDouble = im2double(grayImg);
localTexture = stdfilt(grayDouble, true(5));

%% Create fruit masks
innerMask = imerode(papayaMask, strel("disk", 3));

if sum(innerMask(:)) < 0.55 * sum(papayaMask(:))
    innerMask = papayaMask;
end

safeInnerMask = imerode(innerMask, strel("disk", 1));

if sum(safeInnerMask(:)) < 0.75 * sum(innerMask(:))
    safeInnerMask = innerMask;
end

totalArea = sum(innerMask(:));

if totalArea == 0
    totalArea = 1;
end

%% Green skin
greenMask = innerMask & ...
    H >= 0.15 & H <= 0.48 & ...
    S > 0.15 & ...
    V > 0.18;

greenMask = bwareaopen(greenMask, 15);

%% Yellow and orange skin
yellowMask = innerMask & ...
    H >= 0.035 & H < 0.19 & ...
    S > 0.14 & ...
    V > 0.30;

yellowMask = bwareaopen(yellowMask, 15);

%% Healthy bright orange skin
healthyOrangeMask = safeInnerMask & ...
    H >= 0.025 & H < 0.18 & ...
    S > 0.16 & ...
    V >= 0.43;

healthyOrangeMask = bwareaopen(healthyOrangeMask, 15);

%% Brown damaged skin
% PERUBAHAN: Diluaskan V (ke 0.50) dan diturunkan syarat tekstur untuk tangkap lebam coklat cair
brownCandidate = safeInnerMask & ...
    H >= 0.005 & H < 0.10 & ...
    S > 0.32 & ...
    V >= 0.08 & V < 0.50; 

brownMask = brownCandidate & ~healthyOrangeMask;

brownMask = brownMask & ...
    grayImg < 140 & ...
    localTexture > 0.012; % Diturunkan dari 0.018 untuk lebih peka

brownMask = bwareaopen(brownMask, 60);
brownMask = imopen(brownMask, strel("disk", 2));
brownMask = imclose(brownMask, strel("disk", 1));

%% Dark or black damaged patches
% PERUBAHAN: Tambah syarat tekstur. Tompok gelap mesti kasar (bukan bayang-bayang licin) 
% ATAU ia terlampau hitam mati (V < 0.20)
darkCandidate = safeInnerMask & ...
    ((V < 0.20 & grayImg < 80) | (V < 0.35 & grayImg < 120 & localTexture > 0.015));

normalGreenMask = safeInnerMask & ...
    H >= 0.18 & H <= 0.45 & ...
    S > 0.25 & ...
    V > 0.18;

darkMask = darkCandidate & ~normalGreenMask;

darkMask = bwareaopen(darkMask, 25);
darkMask = imopen(darkMask, strel("disk", 1));
darkMask = imclose(darkMask, strel("disk", 1));

%% White mold
whiteMask = safeInnerMask & ...
    S < 0.18 & ...
    V > 0.74 & ...
    grayImg > 175 & ...
    localTexture > 0.022;

whiteMask = bwareaopen(whiteMask, 45);
whiteMask = imopen(whiteMask, strel("disk", 1));
whiteMask = imclose(whiteMask, strel("disk", 1));

%% Brown and dark lesion mask
lesionMask = (brownMask | darkMask) & safeInnerMask;

lesionMask = bwareaopen(lesionMask, 40);
lesionMask = imclose(lesionMask, strel("disk", 1));

%% Rough texture
roughMask = safeInnerMask & localTexture > 0.065;

roughMask = bwareaopen(roughMask, 35);
roughMask = imopen(roughMask, strel("disk", 1));

%% Combined damage mask
damageMask = brownMask | darkMask | whiteMask;
damageMask = bwareaopen(damageMask, 60);
damageMask = imopen(damageMask, strel("disk", 1));
damageMask = imclose(damageMask, strel("disk", 1));
damageMask = damageMask & safeInnerMask;

%% Percentages
features.greenPercentage = sum(greenMask(:)) / totalArea * 100;
features.yellowPercentage = sum(yellowMask(:)) / totalArea * 100;
features.brownPercentage = sum(brownMask(:)) / totalArea * 100;
features.darkPercentage = sum(darkMask(:)) / totalArea * 100;
features.whiteMoldPercentage = sum(whiteMask(:)) / totalArea * 100;
features.lesionPercentage = sum(lesionMask(:)) / totalArea * 100;
features.roughPercentage = sum(roughMask(:)) / totalArea * 100;
features.damagePercentage = sum(damageMask(:)) / totalArea * 100;
features.damagePercentage = min(max(features.damagePercentage, 0), 100);
features.healthyPercentage = 100 - features.damagePercentage;

%% Connected-region analysis
brownCC = bwconncomp(brownMask);
darkCC = bwconncomp(darkMask);
whiteCC = bwconncomp(whiteMask);
damageCC = bwconncomp(damageMask);
lesionCC = bwconncomp(lesionMask);

features.brownRegionCount = brownCC.NumObjects;
features.darkRegionCount = darkCC.NumObjects;
features.whiteMoldRegionCount = whiteCC.NumObjects;
features.damageRegionCount = damageCC.NumObjects;
features.lesionRegionCount = lesionCC.NumObjects;

%% Largest damaged region
damageStats = regionprops(damageCC, "Area");
if isempty(damageStats)
    features.largestDamagePercentage = 0;
else
    damageAreas = [damageStats.Area];
    features.largestDamagePercentage = max(damageAreas) / totalArea * 100;
end

%% Largest lesion region
lesionStats = regionprops(lesionCC, "Area");
if isempty(lesionStats)
    features.largestLesionPercentage = 0;
else
    lesionAreas = [lesionStats.Area];
    features.largestLesionPercentage = max(lesionAreas) / totalArea * 100;
end

%% Masks for display
features.innerPapayaMask = innerMask;
features.safeInnerMask = safeInnerMask;
features.greenMask = greenMask;
features.yellowMask = yellowMask;
features.healthyOrangeMask = healthyOrangeMask;
features.brownMask = brownMask;
features.darkMask = darkMask;
features.whiteMask = whiteMask;
features.lesionMask = lesionMask;
features.roughMask = roughMask;
features.damageMask = damageMask;

end