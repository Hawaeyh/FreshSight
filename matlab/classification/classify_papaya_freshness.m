function quality = classify_papaya_freshness(features, segmentationQuality)
%CLASSIFY_PAPAYA_FRESHNESS Supporting evidence rules for development review.
% Rule order:
% 1 reliability gate (hard); 2 clustered mold/severe connected damage (hard);
% 3 supported necrosis/lesion combinations (hard); 4 colour/evidence scores;
% 5 clean-yellow rescue. Thresholds are documented train/validation starting
% points and must not be selected from held-out evaluation performance.
statusName = string(segmentationQuality.status);
if statusName == "poor" || statusName == "failed" || ~features.damageReliability
    quality.status = "Unavailable"; quality.grade = "Unavailable"; quality.freshnessScore = NaN;
    quality.suggestion = "Papaya segmentation was unreliable. Capture a clearer image and request manual verification.";
    quality.reliability = "unreliable"; quality.reliabilityScore = segmentationQuality.score;
    quality.reasons = {"Papaya segmentation was unreliable."};
    quality.scores.unripe = NaN; quality.scores.fresh = NaN; quality.scores.rotten = NaN;
    quality.ruleEvidence = struct(); return
end

green = features.greenPercentage; yellow = features.yellowPercentage; orange = features.orangePercentage;
brown = features.brownPercentage; dark = features.darkPercentage; mold = features.whiteMoldPercentage;
damage = features.damagePercentage; lesion = features.lesionPercentage; abnormal = features.roughPercentage;
largestDamage = features.largestDamagePercentage; largestLesion = features.largestLesionPercentage;
damageRegions = features.damageRegionCount;

clusteredMoldGate = mold >= 3.0 && largestDamage >= 2.0;
severeConnectedGate = damage >= 25.0 && largestDamage >= 8.0;
supportedNecrosisGate = dark >= 5.0 && largestDamage >= 4.0 ...
    && (brown >= 3.0 || lesion >= 1.0 || abnormal >= 4.0);
supportedLesionGate = lesion >= 8.0 && largestLesion >= 4.0;

unripeScore = 1.05 * green - 0.45 * (yellow + orange) - 0.35 * damage;
freshScore = 0.80 * yellow + 0.65 * orange - 0.25 * green ...
    - 0.65 * dark - 1.20 * mold - 0.35 * damage;
rottenScore = 0.80 * dark + 1.80 * mold + 0.55 * brown + 0.70 * lesion ...
    + 0.35 * abnormal + 0.55 * largestDamage + 0.04 * min(damageRegions,20);
reasons = strings(0,1);

if clusteredMoldGate
    resultClass = "Rotten"; reasons(end+1) = "Clustered mold with connected damage hard gate.";
elseif severeConnectedGate
    resultClass = "Rotten"; reasons(end+1) = "Severe connected damage hard gate.";
elseif supportedNecrosisGate
    resultClass = "Rotten"; reasons(end+1) = "Dark decay supported by connected colour or texture evidence.";
elseif supportedLesionGate
    resultClass = "Rotten"; reasons(end+1) = "Connected lesion evidence hard gate.";
elseif green >= 65 && damage < 12 && mold < 1.5
    resultClass = "Unripe"; reasons(end+1) = "Predominantly green fruit with limited supported damage.";
else
    [~,index] = max([unripeScore,freshScore,rottenScore]); names = ["Unripe","Fresh","Rotten"];
    resultClass = names(index); reasons(end+1) = "Combined colour and damage-evidence score.";
end

if resultClass == "Rotten" && (yellow + orange) >= 78 && dark < 3 ...
        && mold < 1 && lesion < 3 && largestDamage < 6 && damage < 12
    resultClass = "Fresh"; reasons(end+1) = "Clean ripe-colour rescue with low connected evidence.";
end

switch resultClass
    case "Fresh", grade = "Grade A"; suggestion = "Papaya appears ripe; visually inspect highlighted evidence before handling.";
    case "Unripe", grade = "Grade C"; suggestion = "Papaya is predominantly green and may require further ripening.";
    otherwise, grade = "Grade D"; suggestion = "Supporting MATLAB evidence indicates connected spoilage; compare with the primary AI result and inspect manually.";
end
freshnessScore = max(0,min(100,100 - 0.30*damage - 0.65*dark - 1.10*mold - 0.25*largestLesion));
quality.status = resultClass; quality.grade = grade; quality.freshnessScore = freshnessScore; quality.suggestion = suggestion;
quality.scores.unripe = unripeScore; quality.scores.fresh = freshScore; quality.scores.rotten = rottenScore;
quality.ruleEvidence.clusteredMoldGate = clusteredMoldGate;
quality.ruleEvidence.severeConnectedGate = severeConnectedGate;
quality.ruleEvidence.supportedNecrosisGate = supportedNecrosisGate;
quality.ruleEvidence.supportedLesionGate = supportedLesionGate;
quality.ruleEvidence.damageSeverity = char(features.damageSeverity);
if statusName == "good", quality.reliability = "reliable"; quality.reliabilityScore = segmentationQuality.score;
else, quality.reliability = "low"; quality.reliabilityScore = 0.75*segmentationQuality.score; reasons(end+1) = "Segmentation quality is acceptable but uncertain."; end
quality.reasons = cellstr(reasons);
end
