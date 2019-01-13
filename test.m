%% Pre-process dataset
size = 20;
processed_dataset = pre_process(size);

%% Get various kinds of features
[nist_feat, nist_profile, nist_pix, nist_dis] = feature_generation(processed_dataset);

%% Split into training and test
[trn, tst] = gendat(processed_dataset, 0.5);
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