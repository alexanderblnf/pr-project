%% Pre-process dataset
size = 20;
processed_dataset = pre_process(size);

%% Get various kinds of features
[nist_feat, nist_profile, nist_pix, nist_dis] = feature_generation(processed_dataset);

%% Split into training and test
[trn_feat, tst_feat] = gendat(nist_feat, 0.5);
[trn_profile, tst_profile] = gendat(nist_profile, 0.5);
[trn_pix, tst_pix] = gendat(nist_pix, 0.5);
[trn_dis, tst_dis] = gendat(nist_dis, 0.5);
%%
[errC, optimal_features] = feature_selection(32, trn, tst, ldc);

%%
plote(errC);

%%
for err = errC
    disp(err{:}.error);
end

%%
[w_i, r_i] = featseli(trn, 'eucl-m', 32);

%%
evaluations = evaluations_feat;
for i = 1 : length(evaluations)
    disp(evaluations{:, i});
end