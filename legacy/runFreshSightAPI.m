function jsonResult = runFreshSightAPI(imagePath)
%RUNFRESHSIGHTAPI MATLAB backend entry point for the Flask dashboard.

if nargin < 1 || strlength(string(imagePath)) == 0
    error("FreshSightAPI:MissingImagePath", "An image path is required.");
end

projectRoot = fileparts(mfilename("fullpath"));
addpath(fullfile(projectRoot, "functions"));

img = imread(char(imagePath));
result = analyzePapayaNew(img);

apiResult.Status = char(result.quality.status);
apiResult.Grade = char(result.quality.grade);
apiResult.FreshnessScore = result.quality.freshnessScore;
apiResult.DamagePercentage = result.features.damagePercentage;
apiResult.HealthyPercentage = result.features.healthyPercentage;

apiResult.images.original = imgToBase64(result.processed.rgb);
apiResult.images.papayaMask = imgToBase64(result.papayaMask);
apiResult.images.greenArea = imgToBase64(result.features.greenMask);
apiResult.images.yellowArea = imgToBase64(result.features.yellowMask);
apiResult.images.brownArea = imgToBase64(result.features.brownMask);
apiResult.images.whiteMold = imgToBase64(result.features.whiteMask);
apiResult.images.darkArea = imgToBase64(result.features.darkMask);
apiResult.images.damageHighlight = imgToBase64(result.highlighted);

jsonResult = jsonencode(apiResult);

end

function base64String = imgToBase64(imgMatrix)
%IMGTOBASE64 Convert a MATLAB image or mask matrix into a PNG Base64 string.

if islogical(imgMatrix)
    imgMatrix = uint8(imgMatrix) * 255;
end

tempFile = [tempname, '.png'];
fid = -1;

try
    imwrite(imgMatrix, tempFile);

    fid = fopen(tempFile, "r");
    if fid == -1
        error("FreshSightAPI:FileReadError", "Unable to read temporary PNG file.");
    end

    bytes = fread(fid, Inf, "*uint8");
    fclose(fid);
    fid = -1;

    base64String = matlab.net.base64encode(bytes);
    delete(tempFile);
catch err
    if fid ~= -1
        fclose(fid);
    end

    if isfile(tempFile)
        delete(tempFile);
    end

    rethrow(err);
end

end
