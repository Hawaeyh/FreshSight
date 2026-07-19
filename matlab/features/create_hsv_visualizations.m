function analysis = create_hsv_visualizations(img, papayaMask)
%CREATE_HSV_VISUALIZATIONS Masked HSV channels and composite.
hsvImg = rgb2hsv(img); channels = cell(1, 3);
for index = 1:3
    channel = hsvImg(:, :, index); channel(~papayaMask) = 0; channels{index} = uint8(channel * 255);
end
composite = hsvImg;
for index = 1:3, plane = composite(:, :, index); plane(~papayaMask) = 0; composite(:, :, index) = plane; end
rgbComposite = im2uint8(hsv2rgb(composite)); rgbComposite(repmat(~papayaMask, 1, 1, 3)) = 255;
analysis.hueChannel = channels{1}; analysis.saturationChannel = channels{2}; analysis.valueChannel = channels{3}; analysis.hsvVisualization = rgbComposite;
end
