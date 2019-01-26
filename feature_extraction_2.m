function [err, std, n] = feature_extraction_2(dataset, classifier, variance)

[w, n] = dataset * pcam([], variance);
[err, std] = prcrossval(dataset, w * classifier, 10, 5);

end