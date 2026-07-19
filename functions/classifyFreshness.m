function quality = classifyFreshness(features)
%CLASSIFYFRESHNESS Classify papaya using observed dataset ranges (3 Classes).

%% Read features
green  = features.greenPercentage;
yellow = features.yellowPercentage;
brown  = features.brownPercentage;
dark   = features.darkPercentage;
white  = features.whiteMoldPercentage;
damage = features.damagePercentage;

lesion = features.lesionPercentage;
rough  = features.roughPercentage;

largestDamage = features.largestDamagePercentage;
largestLesion = features.largestLesionPercentage;
damageRegions = features.damageRegionCount;

%% Scores retained for dataset reporting
unripeScore = ...
    green ...
    - 0.65 * yellow ...
    - 0.40 * damage;

freshScore = ...
    1.10 * yellow ...
    - 0.30 * green ...
    - 0.80 * dark ...  % Penalti dinaikkan supaya tompok gelap mengurangkan skor Fresh
    - 1.50 * white ... 
    - 0.30 * damage;   % Penalti kerosakan dinaikkan sikit

rottenScore = ...
    1.20 * dark ...    % Pemberat dark dinaikkan (dari 1.00)
    + 2.00 * white ...
    + 0.50 * damage ...  % Pemberat damage dinaikkan (dari 0.40)
    + 0.60 * brown ...
    + 0.65 * largestDamage ... % Pemberat saiz kerosakan terbesar dinaikkan
    + 0.85 * largestLesion ...
    + 0.20 * rough ...
    + 0.05 * min(damageRegions, 20);

%% 1. Unripe
if green >= 70 && ...
   yellow < 25 && ...
   damage < 10
    status = "Unripe";

%% 2. Strong rotten evidence
% AMBANG DITURUNKAN: Supaya buah yang busuk tak terlepas
elseif white >= 3.0 || ...        % (Dari 4.0)
       dark >= 8.0 || ...         % (Dari 10.0)
       largestLesion >= 12.0 || ... % (Dari 15.0)
       largestDamage >= 15.0 || ... % (Dari 18.0)
       damage >= 25.0             % (Dari 35.0)
    status = "Rotten";

%% 3. Predominantly yellow fruit
% AMBANG DITURUNKAN: Betik kuning yang lebam akan lebih cepat ditangkap sebagai busuk
elseif yellow >= 70 && green < 25
    if dark >= 5.5 % (Dari 6.5)
        status = "Rotten";
    elseif dark >= 3.5 && ... % (Dari 4.0)
           (largestLesion >= 6.0 || ... % (Dari 8.0)
            largestDamage >= 10.0 || ... % (Dari 12.0)
            white >= 1.5 || ... % (Dari 2.0)
            damage >= 20.0) % (Dari 25.0)
        status = "Rotten";
    else
        status = "Fresh";
    end

%% 4. Remaining uncertain fruit
else
    scores = [ ...
        unripeScore, ...
        freshScore, ...
        rottenScore];
    [~, index] = max(scores);
    classNames = ["Unripe", "Fresh", "Rotten"];
    status = classNames(index);
end

%% Safety rule: strong mold or severe dark decay
% Keselamatan diperketatkan
if white >= 5 || ... % (Dari 6)
   (dark >= 8 && largestDamage >= 6) % (Dari 10 dan 8)
    status = "Rotten";
end

%% Safety rule: clean, strongly yellow fruit
if status == "Rotten" && ...
   yellow >= 90 && ...
   dark < 4.0 && ...
   white < 1.5 && ...
   largestLesion < 6 && ...
   largestDamage < 10
    status = "Fresh";
end

%% Output
switch status
    case "Fresh"
        grade = "Grade A";
        suggestion = "Papaya is ripe and suitable for consumption or sale.";
    case "Unripe"
        grade = "Grade C";
        suggestion = "Papaya is predominantly green. Allow it to ripen before consumption.";
    case "Rotten"
        grade = "Grade D";
        suggestion = "Dark decay, mold, or concentrated damaged areas were detected. Not recommended for consumption.";
end

freshnessScore = ...
    100 ...
    - 0.25 * damage ...
    - 0.80 * dark ...
    - 1.00 * white ...
    - 0.20 * largestLesion;

freshnessScore = max(0, min(100, freshnessScore));

quality.status = status;
quality.grade = grade;
quality.freshnessScore = freshnessScore;
quality.suggestion = suggestion;

quality.scores.unripe = unripeScore;
quality.scores.fresh = freshScore;
quality.scores.rotten = rottenScore;

end