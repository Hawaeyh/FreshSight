function background = remove_papaya_background(img, mask, cfg)
%REMOVE_PAPAYA_BACKGROUND Preserve selected fruit on neutral background.
if nargin < 3, cfg = segmentation_config(); end
if size(img, 3) == 1, img = repmat(img, 1, 1, 3); end
background = repmat(cast(cfg.neutralBackgroundValue, 'like', img), size(img));
for channel = 1:3
    plane = background(:, :, channel); source = img(:, :, channel);
    plane(mask) = source(mask); background(:, :, channel) = plane;
end
end
