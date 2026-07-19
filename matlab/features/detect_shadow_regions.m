function shadow = detect_shadow_regions(img, innerMask, localTexture, cfg)
%DETECT_SHADOW_REGIONS Find smooth low-light regions with preserved chroma.
if nargin < 4, cfg = damage_config(); end
hsvImg = rgb2hsv(img); labImg = rgb2lab(img);
S = hsvImg(:,:,2); V = hsvImg(:,:,3); L = labImg(:,:,1) / 100;
gray = im2double(rgb2gray(img)); localContrast = rangefilt(gray, true(5));
validV = V(innerMask); validL = L(innerMask);
if isempty(validV), medianV = 0; medianL = 0; else, medianV = median(validV); medianL = median(validL); end
smoothLowLight = V < medianV * cfg.shadowValueRatio & L < medianL * cfg.shadowLightnessRatio ...
    & localTexture < cfg.shadowMaximumTexture & localContrast < cfg.shadowMaximumLocalContrast;
preservedChromaticity = S > 0.12;
mask = innerMask & smoothLowLight & preservedChromaticity;
mask = imopen(mask, strel('disk',2)); mask = imclose(mask, strel('disk',4));
mask = bwareaopen(mask, max(10, round(nnz(innerMask)*0.001)));
percentage = 100 * nnz(mask) / max(nnz(innerMask),1);
shadow.mask = mask; shadow.percentage = percentage;
shadow.confidence = min(1, percentage / 12 + mean((localTexture(mask) < cfg.shadowMaximumTexture),'all','omitnan') * 0.4);
if isnan(shadow.confidence), shadow.confidence = 0; end
end
