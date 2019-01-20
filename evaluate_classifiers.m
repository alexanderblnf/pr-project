function evaluations = evaluate_classifiers(dataset, classifiers, names, reps, is_dis)
    evaluations = {};
   
    for i = 1: length(classifiers)
        if is_dis == true
            [err, std] = crossvald(dataset, classifiers{:, i}, 10, [], reps);
        else
            [err, std] = prcrossval(dataset, classifiers{:, i}, 10, reps);
        end
        eval = struct;
        eval.name = names{:, i};
        eval.error = err;
        eval.std = std;
        evaluations{end + 1} = eval;
    end
end