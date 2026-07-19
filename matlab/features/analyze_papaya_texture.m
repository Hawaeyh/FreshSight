function texture = analyze_papaya_texture(img, papayaMask)
%ANALYZE_PAPAYA_TEXTURE Masked local texture, edges, roughness and GLCM.
gray = im2double(rgb2gray(img)); localStd = stdfilt(gray, true(5)); localEntropy = entropyfilt(gray, true(9));
edgeMap = edge(gray, 'Canny') & papayaMask; roughMask = papayaMask & (localStd > 0.055 | localEntropy > 5.2);
roughMask = bwareaopen(roughMask, max(12, round(nnz(papayaMask) * 0.0005)));
displayMap = mat2gray(localStd); displayMap(~papayaMask) = 0;
texture.textureMap = im2uint8(displayMap); texture.edgeMap = edgeMap; texture.roughAreaMask = roughMask;
texture.meanEntropy = mean(localEntropy(papayaMask), 'omitnan');
texture.edgeDensity = 100 * nnz(edgeMap) / max(nnz(papayaMask), 1);
texture.roughPercentage = 100 * nnz(roughMask) / max(nnz(papayaMask), 1);
maskedGray = im2uint8(gray); validValues = maskedGray(papayaMask);
try
    % Use only inner-fruit pixels; no artificial black background enters GLCM.
    glcm = graycomatrix(reshape(validValues, 1, []), 'Offset', [0 1], 'Symmetric', true);
    props = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
    texture.glcmContrast = mean(props.Contrast); texture.glcmCorrelation = mean(props.Correlation);
    texture.glcmEnergy = mean(props.Energy); texture.glcmHomogeneity = mean(props.Homogeneity);
catch
    texture.glcmContrast = NaN; texture.glcmCorrelation = NaN; texture.glcmEnergy = NaN; texture.glcmHomogeneity = NaN;
end
end
