function evaluations = evaluate_classifiers(dataset, classifiers, names)
    evaluations = {};
   
    for i = 1: length(classifiers)
        [err, std] = prcrossval(dataset, classifiers{:, i}, 10, 2);
        eval = struct;
        eval.name = names{:, i};
        eval.error = err;
        eval.std = std;
        evaluations{end + 1} = eval;
    end
end