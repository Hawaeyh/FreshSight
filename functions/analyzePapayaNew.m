function result = analyzePapayaNew(img)

tic;

processed = preprocessImage(img);

papayaMask = segmentPapaya(processed.rgb);

features = extractPapayaFeatures(processed.rgb, papayaMask);

quality = classifyFreshness(features);

highlighted = createHighlight(processed.rgb, features.damageMask);

result.original = img;
result.processed = processed;
result.papayaMask = papayaMask;
result.features = features;
result.quality = quality;
result.highlighted = highlighted;
result.processingTime = toc;

end