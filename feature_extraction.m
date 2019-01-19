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

function eval = feature_extraction(dataset, classifier)
    [M, N] = size(dataset);
    index = 1;
    eval = struct;
    while(index <= N)
        m = floor((index + N) / 2);
        disp(m);

        W = dataset * pcam([], m);
        [err_m, std_m] = prcrossval(dataset, W * classifier, 10, 2);

        W = dataset * pcam([], N);
        [err_N, std_N] = prcrossval(dataset, W * classifier, 10, 2);

        if index == N
            eval.error = err_m;
            eval.std = std_m;
            eval.dimension = m;
        end

        if err_m < err_N
            N = m - 1;
            eval.error = err_m;
            eval.std = std_m;
            eval.dimension = m;
        elseif err_m > err_N
            index = m + 1;
            eval.error = err_m;
            eval.std = std_m;
            eval.dimension = m;
        end
    end
end
