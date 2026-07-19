function features = extract_papaya_features(img, segmentation, colour, texture, cfg)
%EXTRACT_PAPAYA_FEATURES Build separate, reliability-gated damage evidence.
% Segmentation behavior is consumed unchanged; all evidence stays in its inner mask.
if nargin < 5, cfg = segmentation_config(); end
damageCfg = damage_config(); inner = segmentation.papayaMaskInner; validArea = nnz(inner);
hsvImg = rgb2hsv(img); V = hsvImg(:,:,3); gray = im2double(rgb2gray(img));
localTexture = stdfilt(gray, true(5)); localContrast = im2double(rangefilt(gray, true(5)));

stem = detect_papaya_stem(img, inner, damageCfg);
shadow = detect_shadow_regions(img, inner, localTexture, damageCfg);
reflection = detect_specular_highlights(img, inner, localTexture, damageCfg);
excluded = stem.mask | shadow.mask | reflection.mask | ~inner;
minimumEvidence = max(12, round(validArea * damageCfg.minimumEvidenceRegionFraction));

lesionSupport = imdilate(colour.masks.redLesion, strel('disk',2));
brownSupport = imdilate(colour.masks.brown, strel('disk',2));
textured = localTexture >= damageCfg.minimumDecayTexture;
contrasted = localContrast >= damageCfg.minimumDecayLocalContrast;

brownDecay = colour.masks.brown & ~excluded ...
    & (textured | contrasted | lesionSupport | V < 0.34);
darkDecay = colour.masks.dark & ~excluded ...
    & (textured | contrasted | lesionSupport | brownSupport);
whiteMold = colour.masks.whiteMold & ~stem.mask & ~reflection.mask & inner ...
    & localTexture >= damageCfg.minimumDecayTexture;
lesion = colour.masks.redLesion & ~stem.mask & ~reflection.mask & inner ...
    & (textured | contrasted | imdilate(brownDecay | darkDecay, strel('disk',2)));
abnormalTexture = texture.roughAreaMask & ~excluded ...
    & (contrasted | imdilate(brownDecay | darkDecay | whiteMold | lesion, strel('disk',3)));

brownDecay = bwareaopen(brownDecay, minimumEvidence);
darkDecay = bwareaopen(darkDecay, minimumEvidence);
whiteMold = bwareaopen(whiteMold, max(12, round(validArea * damageCfg.minimumMoldRegionFraction)));
lesion = bwareaopen(lesion, max(10, round(validArea * damageCfg.minimumLesionRegionFraction)));
abnormalTexture = bwareaopen(abnormalTexture, max(12, round(validArea * damageCfg.minimumTextureRegionFraction)));

combinedEvidence = brownDecay | darkDecay | whiteMold | lesion | abnormalTexture;
combinedEvidence = combinedEvidence & inner & ~stem.mask & ~shadow.mask & ~reflection.mask;
filteredDamage = bwareaopen(combinedEvidence, minimumEvidence);
filteredDamage = imopen(filteredDamage, strel('disk',1));
filteredDamage = imclose(filteredDamage, strel('disk',2)); filteredDamage = filteredDamage & inner;

features.innerPapayaMask = inner; features.safeInnerMask = inner;
features.greenMask = colour.masks.green; features.yellowMask = colour.masks.yellow;
features.healthyOrangeMask = colour.masks.orange; features.brownMask = brownDecay;
features.darkMask = darkDecay; features.whiteMask = whiteMold;
features.lesionMask = lesion; features.roughMask = abnormalTexture;
features.stemMask = stem.mask; features.shadowMask = shadow.mask; features.reflectionMask = reflection.mask;
features.brownDecayMask = brownDecay; features.darkDecayMask = darkDecay;
features.whiteMoldMask = whiteMold; features.abnormalTextureMask = abnormalTexture;
features.combinedDamageEvidence = combinedEvidence;
features.rawDamageMask = combinedEvidence; features.filteredDamageMask = filteredDamage; features.damageMask = filteredDamage;

reliable = segmentation.quality.reliable && validArea > 0;
features.greenPercentage = colour.percentage.green; features.yellowPercentage = colour.percentage.yellow;
features.orangePercentage = colour.percentage.orange; features.unclassifiedPercentage = colour.percentage.unclassified;
features.brownPercentage = percentage(brownDecay, validArea); features.darkPercentage = percentage(darkDecay, validArea);
features.whiteMoldPercentage = percentage(whiteMold, validArea); features.lesionPercentage = percentage(lesion, validArea);
features.roughPercentage = percentage(abnormalTexture, validArea);

damageCC = bwconncomp(filteredDamage); damageStats = regionprops(damageCC,'Area');
features.damageRegionCount = damageCC.NumObjects;
if isempty(damageStats), features.largestDamagePercentage = 0;
else, features.largestDamagePercentage = 100 * max([damageStats.Area]) / max(validArea,1); end
lesionCC = bwconncomp(lesion); lesionStats = regionprops(lesionCC,'Area'); features.lesionRegionCount = lesionCC.NumObjects;
if isempty(lesionStats), features.largestLesionPercentage = 0;
else, features.largestLesionPercentage = 100 * max([lesionStats.Area]) / max(validArea,1); end
brownCC = bwconncomp(brownDecay); darkCC = bwconncomp(darkDecay); moldCC = bwconncomp(whiteMold);
features.brownRegionCount = brownCC.NumObjects;
features.darkRegionCount = darkCC.NumObjects;
features.whiteMoldRegionCount = moldCC.NumObjects;

evidence.brownDecayPercentage = percentage(brownDecay,validArea);
evidence.darkDecayPercentage = percentage(darkDecay,validArea);
evidence.moldPercentage = percentage(whiteMold,validArea);
evidence.lesionPercentage = percentage(lesion,validArea);
evidence.abnormalTexturePercentage = percentage(abnormalTexture,validArea);
evidence.excludedStemPercentage = stem.areaPercentage;
evidence.excludedShadowPercentage = shadow.percentage;
evidence.excludedReflectionPercentage = reflection.percentage;
evidence.stemDetected = stem.detected; evidence.stemConfidence = stem.confidence;
evidence.shadowConfidence = shadow.confidence;
features.damageEvidence = evidence;

if reliable
    features.damagePercentage = min(100,max(0,percentage(filteredDamage,validArea)));
    features.healthyPercentage = 100 - features.damagePercentage;
else
    unavailable = NaN; fields = {'greenPercentage','yellowPercentage','orangePercentage','brownPercentage','darkPercentage','whiteMoldPercentage','lesionPercentage','roughPercentage','unclassifiedPercentage','damagePercentage','healthyPercentage','largestDamagePercentage','largestLesionPercentage'};
    for index = 1:numel(fields), features.(fields{index}) = unavailable; end
    evidenceFields = fieldnames(features.damageEvidence);
    for index = 1:numel(evidenceFields)
        if isnumeric(features.damageEvidence.(evidenceFields{index}))
            features.damageEvidence.(evidenceFields{index}) = unavailable;
        end
    end
end

if ~reliable, severity = "Unavailable";
elseif features.damagePercentage < damageCfg.severityMinimalMaximum, severity = "Minimal";
elseif features.damagePercentage < damageCfg.severityMildMaximum, severity = "Mild";
elseif features.damagePercentage < damageCfg.severityModerateMaximum, severity = "Moderate";
elseif features.largestDamagePercentage >= damageCfg.severeLargestRegionMinimum ...
        || features.whiteMoldPercentage >= 3 || features.lesionPercentage >= 8, severity = "Severe";
else, severity = "Moderate"; end
features.damageSeverity = severity; features.damageReliability = reliable;
features.texture = texture; features.colour = colour;
end

function value = percentage(mask, denominator)
value = min(100,max(0,100 * nnz(mask) / max(denominator,1)));
end
