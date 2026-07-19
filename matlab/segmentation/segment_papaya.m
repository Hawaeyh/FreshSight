function segmentation = segment_papaya(img, cfg)
%SEGMENT_PAPAYA Rank every candidate, refine it, and suppress touching leaves.
% The quality gate is intentionally unchanged; this function only improves
% candidate generation, component selection, and mask refinement.

if nargin < 2, cfg = segmentation_config(); end
if size(img, 3) == 1, img = repmat(img, 1, 1, 3); end
[rows, cols, ~] = size(img); imageArea = rows * cols;
hsvImg = rgb2hsv(img); labImg = rgb2lab(img); gray = imgaussfilt(im2double(rgb2gray(img)), 1.2);
H = hsvImg(:, :, 1); S = hsvImg(:, :, 2); V = hsvImg(:, :, 3);
a = labImg(:, :, 2); b = labImg(:, :, 3);

% Broad but chromatic fruit evidence. The stricter Lab clause avoids merging
% neutral floors, cloth, sky, and image-frame background into one component.
fruitHue = ((H >= 0.015 & H <= 0.49) | H >= 0.94) & S >= 0.14 & V >= 0.11;
labFruit = S >= 0.18 & V >= 0.13 & (b >= 8 | a >= 6);
candidate = (fruitHue | labFruit) & ~(S < 0.11 & V > 0.64);

% Thin, high-variance green structures are more leaf-like than fruit-like.
greenRange = H >= 0.19 & H <= 0.49 & S > 0.20;
greenVariance = stdfilt(H, true(7));
candidateThickness = bwdist(~candidate);
minimumHalfWidth = max(2, round(min(rows, cols) * cfg.leafMinimumWidthFraction));
touchingLeafPixels = candidate & greenRange & candidateThickness < minimumHalfWidth ...
    & greenVariance > cfg.leafGreenVarianceThreshold;
candidate(touchingLeafPixels) = false;

candidate = imopen(candidate, strel('disk', cfg.openRadius));
candidate = imclose(candidate, strel('disk', cfg.closeRadius));
candidate = bwareaopen(candidate, max(cfg.minimumComponentPixels, ...
    round(imageArea * cfg.minimumCandidateAreaFraction)));
candidate = imfill(candidate, 'holes');

cc = bwconncomp(candidate);
baseMasks = cell(1, cc.NumObjects + 1);
for index = 1:cc.NumObjects
    mask = false(rows, cols); mask(cc.PixelIdxList{index}) = true; baseMasks{index} = mask;
end

% A central active-contour proposal can split the target fruit from a single
% merged component that covers the frame, leaves, or neighbouring fruit.
[xx, yy] = meshgrid(1:cols, 1:rows);
centralSeed = ((xx - (cols + 1) / 2) / (cols * cfg.centralSeedWidthFraction / 2)).^2 + ...
    ((yy - (rows + 1) / 2) / (rows * cfg.centralSeedHeightFraction / 2)).^2 <= 1;
baseMasks{end} = centralSeed;

proposalCount = numel(baseMasks); cleanProposals = cell(1, proposalCount);
rawProposals = cell(1, proposalCount); scores = -inf(1, proposalCount);
proposalStats = repmat(struct('Area', 0, 'Centroid', [NaN NaN], 'Solidity', 0, ...
    'Eccentricity', NaN, 'MajorAxisLength', 0, 'MinorAxisLength', 0, ...
    'BoundingBox', [], 'Extent', 0), 1, proposalCount);
borderFractions = zeros(1, proposalCount);
centre = [cols / 2, rows / 2]; maxDistance = hypot(cols / 2, rows / 2);

for index = 1:proposalCount
    initial = baseMasks{index};
    if nnz(initial) == 0, continue; end
    initial = imopen(initial, strel('disk', cfg.openRadius));
    initial = imclose(initial, strel('disk', cfg.closeRadius));
    initial = imfill(initial, 'holes');
    if nnz(initial) == 0, continue; end
    contourSeed = imerode(initial, strel('disk', 2));
    if nnz(contourSeed) < 0.25 * nnz(initial), contourSeed = initial; end
    try
        refined = activecontour(gray, contourSeed, cfg.activeContourIterations, ...
            'Chan-Vese', 'SmoothFactor', cfg.activeContourSmoothFactor);
    catch
        refined = initial;
    end
    % Prevent contour leakage while allowing a modest correction outside the
    % initial colour component. The central proposal remains self-contained.
    if index == proposalCount
        allowanceRadius = max(8, round(min(rows, cols) * 0.15));
    else
        allowanceRadius = max(4, round(min(rows, cols) * 0.025));
    end
    allowed = imdilate(initial, strel('disk', allowanceRadius));
    refined = refined & allowed;
    refined(touchingLeafPixels) = false;
    if index == proposalCount
        % The fallback contour often contains thin hands, stems, or leaves
        % attached to the central fruit. A scale-aware opening disconnects
        % those protrusions without changing the quality gate.
        fallbackOpenRadius = max(cfg.openRadius, round(min(rows, cols) * 0.018));
        openedFallback = imopen(refined, strel('disk', fallbackOpenRadius));
        if nnz(openedFallback) >= 0.45 * nnz(refined)
            refined = openedFallback;
        end
    end
    refined = imopen(refined, strel('disk', cfg.openRadius));
    refined = imclose(refined, strel('disk', cfg.smoothRadius));
    refined = imfill(refined, 'holes');
    refined = bwareaopen(refined, max(cfg.minimumComponentPixels, ...
        round(imageArea * cfg.minimumCandidateAreaFraction)));
    if any(refined(:)), refined = bwareafilt(refined, 1); else, continue; end

    % Low-saturation mold can open a large notch in an otherwise isolated,
    % non-border fruit. Complete only a plausible convex fruit silhouette;
    % never apply this to border-touching background components.
    preBoundary = bwperim(refined); preBorder = false(rows, cols);
    preBorder([1 end], :) = true; preBorder(:, [1 end]) = true;
    preStats = regionprops(refined, 'Solidity');
    if ~any(preBoundary & preBorder, 'all') && ~isempty(preStats) ...
            && preStats(1).Solidity < cfg.componentSelectionMinimumSolidity
        convexCandidate = bwconvhull(refined);
        convexRatio = nnz(convexCandidate) / max(nnz(refined), 1);
        convexAreaPercentage = 100 * nnz(convexCandidate) / imageArea;
        if convexRatio <= cfg.maximumConvexCompletionRatio ...
                && convexAreaPercentage <= cfg.maximumAcceptedAreaPercentage
            refined = convexCandidate;
        end
    end

    stats = regionprops(refined, 'Area', 'Centroid', 'Solidity', 'Eccentricity', ...
        'MajorAxisLength', 'MinorAxisLength', 'BoundingBox', 'Extent');
    if isempty(stats), continue; end
    stats = stats(1); proposalStats(index) = stats;
    boundary = bwperim(refined); borderBand = false(rows, cols);
    borderBand([1 end], :) = true; borderBand(:, [1 end]) = true;
    borderFraction = 100 * nnz(boundary & borderBand) / max(nnz(boundary), 1);
    borderFractions(index) = borderFraction;
    areaFraction = stats.Area / imageArea;
    areaScore = max(0, 1 - abs(areaFraction - 0.45) / 0.45);
    solidityScore = min(1, max(0, (stats.Solidity - 0.45) / 0.50));
    eccentricityScore = exp(-abs(stats.Eccentricity - 0.68) / 0.38);
    extentScore = exp(-abs(stats.Extent - 0.72) / 0.35);
    borderScore = max(0, 1 - borderFraction / 40);
    centreScore = max(0, 1 - norm(stats.Centroid - centre) / maxDistance);
    aspectRatio = stats.MajorAxisLength / max(stats.MinorAxisLength, eps);
    aspectScore = exp(-abs(log(max(aspectRatio, eps) / 1.50)) / 0.75);
    chromaticFraction = mean(fruitHue(refined));
    leafVarianceFraction = mean(greenVariance(refined & greenRange) > cfg.leafGreenVarianceThreshold, 'all');
    if isnan(leafVarianceFraction), leafVarianceFraction = 0; end
    colourScore = max(0, min(1, chromaticFraction - 0.35 * leafVarianceFraction));
    scores(index) = cfg.weight_area * areaScore + cfg.weight_solidity * solidityScore + ...
        cfg.weight_eccentricity * eccentricityScore + cfg.weight_extent * extentScore + ...
        cfg.weight_border_contact * borderScore + cfg.weight_center_distance * centreScore + ...
        cfg.weight_aspect_ratio * aspectScore + cfg.weight_colour_consistency * colourScore;
    cleanProposals{index} = refined;
    rawProposals{index} = imfill(initial | imdilate(refined, strel('disk', 2)), 'holes');
end

% Prefer a real colour-component proposal whenever its refined geometry
% already satisfies the unchanged gate. The central active contour is a
% fallback for merged/full-frame candidates, not a shortcut to a small core.
qualifiedBase = false(1, max(0, proposalCount - 1));
for index = 1:(proposalCount - 1)
    if ~isfinite(scores(index)), continue; end
    stats = proposalStats(index); areaPercentageCandidate = 100 * stats.Area / imageArea;
    qualifiedBase(index) = areaPercentageCandidate >= cfg.minimumAcceptedAreaPercentage ...
        && areaPercentageCandidate <= cfg.maximumAcceptedAreaPercentage ...
        && borderFractions(index) <= cfg.maximumBorderTouchPercentage ...
        && stats.Solidity >= cfg.componentSelectionMinimumSolidity;
end
if any(qualifiedBase)
    eligibleScores = scores(1:(proposalCount - 1)); eligibleScores(~qualifiedBase) = -inf;
    [selectedScore, selectedIndex] = max(eligibleScores);
else
    centralAreaFraction = proposalStats(proposalCount).Area / imageArea;
    croppedBase = false(1, max(0, proposalCount - 1));
    for index = 1:(proposalCount - 1)
        if ~isfinite(scores(index)), continue; end
        stats = proposalStats(index); areaFraction = stats.Area / imageArea;
        croppedBase(index) = stats.Solidity >= cfg.componentSelectionMinimumSolidity ...
            && areaFraction >= 0.50 && areaFraction <= 0.85 ...
            && borderFractions(index) > cfg.maximumBorderTouchPercentage;
    end
    if centralAreaFraction < 0.35 && any(croppedBase)
        eligibleScores = scores(1:(proposalCount - 1)); eligibleScores(~croppedBase) = -inf;
        [selectedScore, selectedIndex] = max(eligibleScores);
    else
    [selectedScore, selectedIndex] = max(scores);
    end
end
if isempty(selectedIndex) || ~isfinite(selectedScore)
    rawMask = false(rows, cols); cleanMask = rawMask;
    selectedScore = 0; selectedIndex = 0;
else
    rawMask = rawProposals{selectedIndex}; cleanMask = cleanProposals{selectedIndex};
end
innerMask = imerode(cleanMask, strel('disk', cfg.innerErosionRadius));
if nnz(innerMask) < 0.55 * nnz(cleanMask), innerMask = imerode(cleanMask, strel('disk', 2)); end

warnings = strings(0, 1); areaPercentage = 100 * nnz(cleanMask) / imageArea;
borderPercentage = 0; solidity = 0; eccentricity = NaN; aspectRatio = NaN; boundingBox = []; centroid = [];
if selectedIndex > 0
    selected = proposalStats(selectedIndex); solidity = selected.Solidity; eccentricity = selected.Eccentricity;
    aspectRatio = selected.MajorAxisLength / max(selected.MinorAxisLength, eps);
    boundingBox = selected.BoundingBox; centroid = selected.Centroid; borderPercentage = borderFractions(selectedIndex);
end
if selectedIndex == 0, warnings(end+1) = "No plausible papaya component was found."; end
if areaPercentage < cfg.minimumAcceptedAreaPercentage, warnings(end+1) = "Papaya mask is too small."; end
if areaPercentage > cfg.maximumAcceptedAreaPercentage, warnings(end+1) = "Papaya mask occupies too much of the image."; end
if borderPercentage > cfg.maximumBorderTouchPercentage, warnings(end+1) = "Papaya mask touches too much of the image border."; end
if solidity < cfg.minimumSolidity, warnings(end+1) = "Selected component has implausibly low solidity."; end
if ~isnan(aspectRatio) && (aspectRatio < 1.05 || aspectRatio > 4.8), warnings(end+1) = "Selected component has an unusual aspect ratio."; end
componentAreas = cellfun(@nnz, cleanProposals, 'UniformOutput', true); similarCount = 0;
if selectedIndex > 0
    similarCount = nnz(componentAreas >= cfg.similarComponentAreaRatio * componentAreas(selectedIndex));
    if similarCount > cfg.maximumSimilarComponents, warnings(end+1) = "Multiple similarly sized objects were detected."; end
end

% Reliability-gate thresholds and behavior below are preserved verbatim.
hardFailure = selectedIndex == 0 || areaPercentage < 2 || areaPercentage > 95 || nnz(innerMask) == 0;
poor = areaPercentage < cfg.minimumAcceptedAreaPercentage || areaPercentage > cfg.maximumAcceptedAreaPercentage || borderPercentage > cfg.maximumBorderTouchPercentage || solidity < cfg.minimumSolidity;
if hardFailure, status = "failed";
elseif poor || selectedScore < cfg.minimumAcceptableScore, status = "poor";
elseif selectedScore < cfg.minimumGoodScore || ~isempty(warnings), status = "acceptable";
else, status = "good"; end

segmentation.papayaMaskRaw = rawMask; segmentation.papayaMaskClean = cleanMask; segmentation.papayaMaskInner = innerMask;
segmentation.backgroundRemoved = remove_papaya_background(img, cleanMask, cfg); segmentation.transparentBackground = segmentation.backgroundRemoved;
segmentation.boundingBox = boundingBox; segmentation.centroid = centroid; segmentation.score = max(0, min(1, selectedScore));
segmentation.quality.status = status; segmentation.quality.score = segmentation.score;
segmentation.quality.papayaAreaPercentage = areaPercentage; segmentation.quality.borderTouchingPercentage = borderPercentage;
segmentation.quality.solidity = solidity; segmentation.quality.eccentricity = eccentricity; segmentation.quality.aspectRatio = aspectRatio;
segmentation.quality.componentCount = cc.NumObjects; segmentation.quality.similarComponentCount = similarCount;
segmentation.quality.warnings = cellstr(warnings); segmentation.quality.reliable = status == "good" || status == "acceptable";
end
