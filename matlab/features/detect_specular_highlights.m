function reflection = detect_specular_highlights(img, innerMask, localTexture, cfg)
%DETECT_SPECULAR_HIGHLIGHTS Exclude smooth bright low-saturation reflections.
if nargin < 4, cfg = damage_config(); end
hsvImg = rgb2hsv(img); S = hsvImg(:,:,2); V = hsvImg(:,:,3);
mask = innerMask & S < cfg.reflectionMaximumSaturation ...
    & V > cfg.reflectionMinimumValue & localTexture < cfg.reflectionMaximumTexture;
mask = imopen(mask, strel('disk',1)); mask = imclose(mask, strel('disk',2));
mask = bwareaopen(mask, max(6, round(nnz(innerMask)*0.0003)));
reflection.mask = mask; reflection.percentage = 100 * nnz(mask) / max(nnz(innerMask),1);
end
