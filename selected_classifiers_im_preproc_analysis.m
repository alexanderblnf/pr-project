%% Generate datasets
image_size = 20;
dataset_step = 4;
% [processed_dataset, data] = pre_process(dataset_step, image_size);
processed_dataset_simple = simple_preprocess(dataset_step, image_size);
%%
[~, nist_pix, nist_dis, nist_dis_cos] = feature_generation(processed_dataset, true, false);
[nist_feat, ~, ~, ~] = feature_generation(processed_dataset_simple, true, false);

%% Select classifiers
classifiers = {loglc, fisherc, knnc([], 1), knnc([], 3), parzenc, bpxnc([],[30 20],1000)};
names = {'loglc', 'fisherc', '1-NN', '3-NN', 'Parzen', 'bpxnc'};

%% Test feature dataset
evaluations_feat = evaluate_classifiers(nist_feat, classifiers, names);

%% Test profile dataset
evaluations_prof = evaluate_classifiers(nist_profile, classifiers, names, 2);

%% Test pixel dataset
evaluations_pix = evaluate_classifiers(nist_pix, classifiers, names);

%% Classifiers dissimilarity
classifiers = {fisherc, knnc([], 1), knnc([], 3), parzenc, bpxnc([],[26 20],1000)};
names = {'fisherc', '1-NN', '3-NN', 'Parzen', 'bpxnc'};

%% Test dissimilarity dataset
evaluations_dis = evaluate_classifiers(nist_dis, classifiers, names);

%% 
evaluations_dis_cos = evaluate_classifiers(nist_dis_cos, classifiers, names);

%% Classifiers to combine
set1 = {fisherc, knnc([], 1)};
set2 = {fisherc, knnc([], 3)};
set3 = {fisherc, parzenc};
set4 = {knnc([], 1), parzenc};
set5 = {knnc([], 3), parzenc};

%% 
dataset = nist_pix;
classifiers = set1;
[combinedP, combinedS, names] = combined_classifiers(dataset, classifiers);

[errP, stdP] = prcrossval(dataset, combinedP, 10, 1);
[errS, stdS] = prcrossval(dataset, combinedS, 10, 1);

%%
dataset = nist_feat;
classifier = fisherc;

result = feature_extraction_2(dataset, classifier);

%%
dataset = nist_dis_cos;
classifier = knnc([], 3);

W = dataset * pcam([], 10);
[err_N, std_N] = prcrossval(dataset, classifier, 10, 5);
