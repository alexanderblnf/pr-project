function results = feature_extraction_frac(dataset, classifiers)
    results = cell(1, length(classifiers));
    for i = 1 : length(results)
        results{i} = zeros(1, 10);
    end
    
    for i = 0.1 : 0.1 : 0.9
        [w, n] = dataset * pcam([], i);
        disp(n);
        result = prcrossval(dataset, w * classifiers, 10, 2);
        for j = 1 : length(result)
            results{j}(uint8(i * 10)) = result(j);
        end
    end
    
    [w, n] = dataset * pcam([], 0.99);
    result = prcrossval(dataset, w * classifiers, 10, 2);
    for j = 1 : length(result)
        results{j}(10) = result(j);
    end
end