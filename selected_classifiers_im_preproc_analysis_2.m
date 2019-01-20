%% Generate datasets
image_size = 20;
dataset_step = 100;
%%
[small_processed_dataset_simple] = simple_preprocess(dataset_step, image_size);
[nist_feat, nist_profile, ~, ~, ~] = feature_generation(small_processed_dataset_simple, true, true);

%%
[small_processed_dataset] = pre_process(dataset_step, image_size);
[~, ~, nist_pix, nist_dis, nist_dis_cos] = feature_generation(small_processed_dataset, false, false);

%% Select classifiers
classifiers = {ldc, fisherc, knnc([], 1), parzenc, bpxnc([],[30 20],1000)};
names = {'loglc', 'fisherc', '1-NN', '3-NN', 'Parzen', 'bpxnc'};

classifiers = {loglc, fisherc, ldc, naivebc, nmc, qdc, knnc([], 1), knnc([], 2), knnc([], 3), parzenc, bpxnc([],[26 20],1000), perlc([], 1000), dtc([])};
names = {'loglc', 'fisherc', 'ldc','naivebc','nmc', 'qdc', '1-NN', '2-NN', '3-NN', 'Parzen', 'bpxnc', 'perlc', 'dtc'};

%% Test feature dataset
evaluations_feat = evaluate_classifiers(nist_feat, classifiers, names, 5);

%% Test profile dataset
% evaluations_prof = evaluate_classifiers(nist_profile, classifiers, names);

%% Test pixel dataset
evaluations_pix = evaluate_classifiers(nist_pix, classifiers, names, 5);

%% Classifiers dissimilarity
classifiers = {fisherc, ldc, naivebc, nmc, qdc, knnc([], 1), knnc([], 2), knnc([], 3), parzenc, bpxnc([],[80 20],1000), perlc([], 1000), dtc([])};
names = {'fisherc', 'ldc', 'naivebc', 'nmc', 'qdc', '1-NN', '2-NN', '3-NN', 'Parzen', 'bpxnc', 'perlc', 'dtc'};

%% Test dissimilarity dataset
evaluations_dis = evaluate_classifiers(nist_dis, classifiers, names, 5);

%% 
evaluations_dis_cos = evaluate_classifiers(nist_dis_cos, classifiers, names, 5);

%% Classifiers to combine
set1 = {fisherc, knnc([], 1)};
set2 = {fisherc, knnc([], 3)};
set3 = {fisherc, parzenc};
set4 = {knnc([], 1), parzenc};
set5 = {knnc([], 3), parzenc};

%% 
dataset1 = nist_pix;
classifier1 = parzenc;
num_pc = [32];

dataset2 = nist_feat;
classifier2 = ldc;

[combinedP, combinedS, names] = combined_classifiers(dataset1, dataset2, num_pc, classifier1, classifier2);
%%
[errP, stdP] = prcrossval([dataset1, dataset2], combinedP, 10, 1);
%%
[errS, stdS] = prcrossval(dataset, combinedS, 10, 1);

%%
dataset = nist_feat;
classifier = bpxnc([], [30 10], 1000);
[M, N] = size(dataset);

result = feature_extraction_2(dataset, classifier, 1, N);

%%
dataset = nist_pix;
classifier = bpxnc([], [30 10], 1000);

W = dataset * pcam([], 43);
[err_N, std_N] = prcrossval(dataset, W * classifier, 10, 5);

%%

evaluations = evaluations_dis_cos;
for i = 1 : length(evaluations)
    disp(evaluations{:, i});
end
