%% Generate datasets
image_size = 20;
dataset_step = 100;
% [processed_dataset, data] = pre_process(dataset_step, image_size);
processed_dataset_simple = simple_preprocess(dataset_step, image_size);

%%
% [~, ~, nist_pix, nist_dis, nist_dis_cos] = feature_generation(processed_dataset, true, false);
[nist_feat, nist_profile, ~, ~, ~] = feature_generation(processed_dataset_simple, true, true);
%% Test feature dataset
classifiers = {loglc, fisherc, ldc, naivebc, nmc, qdc, knnc([], 1), knnc([], 2), knnc([], 3), parzenc, bpxnc([],[26 20],1000), perlc([], 1000), dtc([])};
names = {'loglc', 'fisherc', 'ldc','naivebc','nmc', 'qdc', '1-NN', '2-NN', '3-NN', 'Parzen', 'bpxnc', 'perlc', 'dtc'};
evaluations_feat = evaluate_classifiers(nist_feat, classifiers, names, 5, false);
%% Test profile dataset
classifiers = {loglc, fisherc,ldc, naivebc, nmc, qdc, knnc([], 1), knnc([], 2), knnc([], 3), parzenc, bpxnc([],[32 20],1000), perlc([], 1000), dtc([])};
names = {'loglc', 'fisherc', 'ldc', 'naivebc', 'nmc', 'qdc', '1-NN', '2-NN', '3-NN', 'Parzen', 'bpxnc', 'perlc', 'dtc'};
evaluations_prof = evaluate_classifiers(nist_profile, classifiers, names, 2, false);

%% Test pixel dataset
classifiers = {loglc, fisherc, ldc, naivebc, nmc, qdc, knnc([], 1), knnc([], 2), knnc([], 3), parzenc, bpxnc([],[32 20],1000), perlc([], 1000), dtc([])};
names = {'loglc', 'fisherc', 'ldc', 'naivebc', 'nmc', 'qdc', '1-NN', '2-NN', '3-NN', 'Parzen', 'bpxnc', 'perlc', 'dtc'};
evaluations_pix = evaluate_classifiers(nist_pix, classifiers, names, 5, false);
%% Test dissimilarity dataset - Euclidean distance
classifiers = {fisherc, ldc, naivebc, nmc, qdc, knndc([], 1), knndc([], 2), knndc([], 3), parzenddc, bpxnc([],[80 20],1000), perlc([], 1000), dtc([])};
names = {'fisherc', 'ldc', 'naivebc', 'nmc', 'qdc', '1-NN', '2-NN', '3-NN', 'Parzen', 'bpxnc', 'perlc', 'dtc'};
evaluations_dis = evaluate_classifiers(nist_dis, classifiers, names, 5, true);

%% Test dissimilarity dataset - Euclidean distance
classifiers = {fisherc, ldc, naivebc, nmc, qdc, knndc([], 1), knndc([], 2), knndc([], 3), parzenddc, bpxnc([],[80 20],1000), perlc([], 1000), dtc([])};
names = {'fisherc', 'ldc', 'naivebc', 'nmc', 'qdc', '1-NN', '2-NN', '3-NN', 'Parzen', 'bpxnc', 'perlc', 'dtc'};
evaluations_dis_cos = evaluate_classifiers(nist_dis_cos, classifiers, names, 5, true);

