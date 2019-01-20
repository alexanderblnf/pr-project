%% Generate datasets
image_size = 20;
dataset_step = 4;
[processed_dataset, data] = pre_process(dataset_step, image_size);

%%
[nist_feat, ~, nist_pix, nist_dis, nist_dis_cos] = feature_generation(processed_dataset, true, false);

%% Split into training and test
[trn_feat, tst_feat] = gendat(nist_feat, 0.5);
[trn_pix, tst_pix] = gendat(nist_pix, 0.5);
[trn_dis, tst_dis] = gendat(nist_dis, 0.5);
[trn_dis_cos, tst_dis_cos] = gendat(nist_dis_cos, 0.5);

%%
classifiers = {loglc, fisherc, knndc([], 1), knndc([], 3), parzenddc, bpxnc([],[26 20],1000)};
names = {'loglc', 'fisherc', '1-NN', '3-NN', 'Parzen', 'bpxnc'};
errC_feat_all = cell(1, 6);
errC_pix_all = cell(1, 6);
errC_dis_all = cell(1, 5);
errC_dis_cos_all = cell(1, 5);

%%
% for i = 1 : length(classifiers)
%     [errC_feat, optimal_features_feat] = feature_selection(24, trn_feat, tst_feat, classifiers{:, i}, 1, false);
%     index = find_best_optimal_features(optimal_features_feat);
%     errC_feat_all{i} = errC_feat{:, index};
%     
%     [errC_pix, optimal_features_pix] = feature_selection(484, trn_pix, tst_pix, classifiers{:, i}, 10, true);
%     errC_pix_all{i} = errC_pix{:, 1};
    
%     if i > 1
        [errC_dis, optimal_features_dis] = feature_selection(2500, nist_dis, tst_dis, classifiers, 50, true, true);
        errC_dis_all{i - 1} = errC_dis{:, 1};

        [errC_dis_cos, optimal_features_dis_cos] = feature_selection(2500, nist_dis_cos, tst_dis_cos, classifiers{:, i}, 50, true, true);
        errC_dis_cos_all{i - 1} = errC_dis_cos{:, 1};
%     end
% end

%%
for i = 1 : length(classifiers)
    if i > 1
        [errC_dis, optimal_features_dis] = feature_selection(1000, trn_dis, tst_dis, classifiers{:, i}, 10, true);
        errC_dis_all{i - 1} = errC_dis{:, 1};

        [errC_dis_cos, optimal_features_dis_cos] = feature_selection(1000, trn_dis_cos, tst_dis_cos, classifiers{:, i}, 10, true);
        errC_dis_cos_all{i - 1} = errC_dis_cos{:, 1};
    end
end

%% Test different distribution for dis
[trn_dis, tst_dis] = gendat(nist_dis, 0.5);

[errC_dis, optimal_features_dis] = feature_selection(1000, trn_dis, tst_dis, bpxnc([],[30 20],1000), 30, true);
errC_dis_all{5} = errC_dis{:, 1};

%%

function index = find_best_optimal_features(optimal_features)
    minError = 1;
    index = 0;
    for i = 1 : length(optimal_features)
        error = optimal_features{:, i}.error;
        if minError > error
            minError = error;
            index = i;
        end
    end
end