function processed = preprocess_papaya_image(img)
%PREPROCESS_PAPAYA_IMAGE Prepare an RGB papaya image for rule-based analysis.

if size(img, 3) == 1
    img = cat(3, img, img, img);
end

processed.original = img;
maxSize = 512;
[h, w, ~] = size(img);

if max(h, w) > maxSize
    scale = maxSize / max(h, w);
    analysisImg = imresize(img, scale);
else
    analysisImg = img;
end

analysisImg = imgaussfilt(analysisImg, 0.5);
labImg = rgb2lab(analysisImg);
L = labImg(:, :, 1) / 100;
LAdjusted = adapthisteq(L, 'NumTiles', [4 4], 'ClipLimit', 0.005);
labImg(:, :, 1) = LAdjusted * 100;
correctedImg = lab2rgb(labImg, 'OutputType', 'uint8');

processed.rgb = correctedImg;
processed.gray = rgb2gray(correctedImg);
processed.hsv = rgb2hsv(correctedImg);
end
