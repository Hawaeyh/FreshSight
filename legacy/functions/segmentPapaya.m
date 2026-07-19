function papayaMask = segmentPapaya(img)
%SEGMENTPAPAYA Segment the papaya from the image background.
%
% Input:
%   img - RGB papaya image
%
% Output:
%   papayaMask - logical binary mask containing the papaya region

% Ensure RGB image
if size(img, 3) == 1
    img = cat(3, img, img, img);
end

% Convert to HSV
hsvImg = rgb2hsv(img);

H = hsvImg(:, :, 1);
S = hsvImg(:, :, 2);
V = hsvImg(:, :, 3);

% ---------------------------------------------------------
% Candidate papaya colours
% ---------------------------------------------------------

% Green skin
greenCandidate = ...
    H >= 0.15 & H <= 0.48 & ...
    S > 0.15 & ...
    V > 0.18;

% Yellow and orange skin
yellowCandidate = ...
    H >= 0.035 & H < 0.18 & ...
    S > 0.15 & ...
    V > 0.22;

% Brown damaged skin
brownCandidate = ...
    H >= 0.015 & H < 0.12 & ...
    S > 0.20 & ...
    V > 0.10;

% Combine papaya-related colours
papayaMask = greenCandidate | yellowCandidate | brownCandidate;

% ---------------------------------------------------------
% Morphological cleaning
% ---------------------------------------------------------

% Remove isolated pixels
papayaMask = bwareaopen(papayaMask, 300);

% Join broken fruit regions
papayaMask = imclose(papayaMask, strel("disk", 10));

% Fill black, rotten and white regions located inside the fruit
papayaMask = imfill(papayaMask, "holes");

% Smooth the mask
papayaMask = imopen(papayaMask, strel("disk", 3));
papayaMask = imclose(papayaMask, strel("disk", 6));

% ---------------------------------------------------------
% Keep only the largest object
% ---------------------------------------------------------

cc = bwconncomp(papayaMask);

if cc.NumObjects > 0
    objectSizes = cellfun(@numel, cc.PixelIdxList);
    [~, largestIndex] = max(objectSizes);

    largestMask = false(size(papayaMask));
    largestMask(cc.PixelIdxList{largestIndex}) = true;

    papayaMask = largestMask;
else
    papayaMask = false(size(H));
end

% Final hole filling and smoothing
papayaMask = imfill(papayaMask, "holes");
papayaMask = imclose(papayaMask, strel("disk", 5));

end