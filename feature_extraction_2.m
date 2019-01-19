% function result = feature_extraction(dataset, stepp, classifier)
% dataset = nist_pix;
% [M, N] = size(dataset);
% 
% stepp = 50;
% 
% total_err = zeros(ceil(N/stepp), 1); 
% evaluations = {};
% 
% for i = 1 : stepp : N
%     disp(i);
%     W = dataset * pcam([], i);
%     [err, std] = prcrossval(dataset, W * fisherc, 10, 2);
% 
%     total_err(ceil(i/stepp)) = err;
%     eval = struct;
%     eval.error = err;
%     eval.std = std;
%     eval.dimension = i;
%     evaluations{end + 1} = eval;
% end
% 
% [min_err, dimension] = min(total_err);
% result = evaluations{:, dimension};

function result = feature_extraction_2(dataset, classifier, step, N)
    all_errors = zeros(1, ceil(N/step));
    evaluations = {};
    
    for i = 1:step:N
        disp(i);
        W = dataset * pcam([], i);
        [err, std] = prcrossval(dataset, W * classifier, 10, 2);
        all_errors(ceil(i / step)) = err;
        eval = struct;
        eval.error = err;
        eval.std = std;
        eval.dimension = i;
        evaluations{end + 1} = eval;
    end
    
    [~, dimension] = min(all_errors);
    result = evaluations{:, dimension};
end
