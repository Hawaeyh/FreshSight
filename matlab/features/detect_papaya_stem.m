function stem = detect_papaya_stem(img, innerMask, cfg)
%DETECT_PAPAYA_STEM Conservative endpoint stem/scar detector.
if nargin < 3, cfg = damage_config(); end
hsvImg = rgb2hsv(img); H = hsvImg(:, :, 1); S = hsvImg(:, :, 2); V = hsvImg(:, :, 3);
validArea = nnz(innerMask); stemMask = false(size(innerMask)); confidence = 0;
fruitStats = regionprops(innerMask, 'BoundingBox');
if validArea > 0 && ~isempty(fruitStats)
    box = fruitStats(1).BoundingBox; yTop = max(1, floor(box(2))); yBottom = min(size(innerMask,1), ceil(box(2)+box(4)));
    bandHeight = max(3, round(box(4) * cfg.stemEndpointBandFraction)); endpointBand = false(size(innerMask));
    endpointBand(yTop:min(yBottom,yTop+bandHeight),:) = true;
    endpointBand(max(yTop,yBottom-bandHeight):yBottom,:) = true;
    brownOrDark = ((H < 0.12 | H > 0.94) & S > 0.18 & V < 0.58) | V < 0.22;
    candidate = innerMask & endpointBand & brownOrDark;
    candidate = imopen(candidate, strel('disk',1));
    candidate = bwareaopen(candidate, max(6, round(validArea * 0.0003)));
    cc = bwconncomp(candidate); stats = regionprops(cc,'Area','Eccentricity','BoundingBox','PixelIdxList');
    bestScore = 0;
    for index = 1:numel(stats)
        areaPercentage = 100 * stats(index).Area / validArea;
        widthFraction = stats(index).BoundingBox(3) / max(box(3),1);
        narrowScore = max(0, 1 - widthFraction / 0.35);
        sizeScore = max(0, 1 - areaPercentage / cfg.stemMaximumAreaPercentage);
        shapeScore = max(stats(index).Eccentricity, narrowScore);
        score = 0.45 * sizeScore + 0.35 * shapeScore + 0.20 * narrowScore;
        if areaPercentage <= cfg.stemMaximumAreaPercentage && score > bestScore
            stemMask(:) = false; stemMask(stats(index).PixelIdxList) = true; bestScore = score;
        end
    end
    confidence = min(1, bestScore);
end
stem.mask = stemMask; stem.detected = any(stemMask(:));
stem.areaPercentage = 100 * nnz(stemMask) / max(validArea,1); stem.confidence = confidence;
end
