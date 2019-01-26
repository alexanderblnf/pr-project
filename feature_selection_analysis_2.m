%% Split into training and test
[trn_feat, tst_feat] = gendat(nist_feat, 0.5);
[trn_pix, tst_pix] = gendat(nist_pix, 0.5);
[trn_profile, tst_profile] = gendat(nist_profile, 0.5);

%%
classifiers = {ldc, fisherc, knnc([], 1), parzenc, nmc};
names = {'ldc', 'fisherc', '1-NN', 'Parzen', 'nmc'};
errC_feat_all = cell(1, 5);
errC_pix_all = cell(1, 5);
errC_prof_all = cell(1, 5);

%%
for i = 1 : length(classifiers)
    [errC_feat, optimal_features_feat] = feature_selection(24, trn_feat, tst_feat, classifiers{:, i}, 1, false);
    index = find_best_optimal_features(optimal_features_feat);
    errC_feat_all{i} = errC_feat{:, index};
    
    [errC_pix, optimal_features_pix] = feature_selection(400, trn_pix, tst_pix, classifiers{:, i}, 10, true);
    errC_pix_all{i} = errC_pix{:, 1};

    [errC_prof, optimal_features_prof] = feature_selection(44, trn_profile, tst_profile, classifiers{:, i}, 2, false);
    index = find_best_optimal_features(optimal_features_prof);
    errC_prof_all{i} = errC_prof{:, index};
end

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