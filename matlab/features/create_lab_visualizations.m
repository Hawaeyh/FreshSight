function analysis = create_lab_visualizations(img, papayaMask)
%CREATE_LAB_VISUALIZATIONS Masked Lab channels and within-fruit statistics.
lab = rgb2lab(img); names = {'L', 'a', 'b'};
for index = 1:3
    values = lab(:, :, index); valid = values(papayaMask); displayChannel = zeros(size(values), 'uint8');
    if ~isempty(valid)
        low = min(valid); high = max(valid); normalized = (values - low) / max(high - low, eps);
        displayChannel(papayaMask) = uint8(255 * normalized(papayaMask)); analysis.mean.(names{index}) = mean(valid); analysis.standardDeviation.(names{index}) = std(valid);
    else, analysis.mean.(names{index}) = NaN; analysis.standardDeviation.(names{index}) = NaN; end
    analysis.([names{index}, 'Channel']) = displayChannel;
end
analysis.labVisualization = lab2rgb(lab, 'OutputType', 'uint8'); analysis.labVisualization(repmat(~papayaMask, 1, 1, 3)) = 255;
end
