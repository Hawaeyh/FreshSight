function processed = preprocessImage(img)

if size(img, 3) == 1
    img = cat(3, img, img, img);
end

processed.original = img;

% Resize only for analysis
maxSize = 512;

[h, w, ~] = size(img);

if max(h, w) > maxSize
    scale = maxSize / max(h, w);
    analysisImg = imresize(img, scale);
else
    analysisImg = img;
end

% Mild filtering
analysisImg = imgaussfilt(analysisImg, 0.5);

% Mild illumination correction
labImg = rgb2lab(analysisImg);
L = labImg(:,:,1) / 100;

LAdjusted = adapthisteq( ...
    L, ...
    'NumTiles', [4 4], ...
    'ClipLimit', 0.005);

labImg(:,:,1) = LAdjusted * 100;

correctedImg = lab2rgb(labImg, 'OutputType', 'uint8');

processed.rgb = correctedImg;
processed.gray = rgb2gray(correctedImg);
processed.hsv = rgb2hsv(correctedImg);

end