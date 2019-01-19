%% Generate datasets
image_size = 20;
dataset_step = 4;
[processed_dataset, data] = pre_process(dataset_step, image_size);
%%
[nist_feat, nist_pix, nist_dis, nist_dis_cos] = feature_generation(processed_dataset);

%% Select classifiers
classifiers = {loglc, fisherc, knnc([], 1), knnc([], 3), parzenc, bpxnc([],[26 20],1000)};
names = {'loglc', 'fisherc', '1-NN', '3-NN', 'Parzen', 'bpxnc'};

%% Test feature dataset
evaluations_feat = evaluate_classifiers(nist_feat, classifiers, names);

%% Test profile dataset
% evaluations_prof = evaluate_classifiers(nist_profile, classifiers, names);

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
dataset = nist_dis_cos;
classifier = knnc([], 3);

result = feature_extraction(dataset, classifier);