%% Generate datasets
image_size = 20;
dataset_step = 2;
processed_dataset = pre_process(2, size);
[nist_feat, nist_profile, nist_pix, nist_dis] = feature_generation(processed_dataset);

%% Test feature dataset
classifiers = {loglc, fisherc, ldc, nmc, qdc, knnc([], 1), knnc([], 2), knnc([], 3), parzenc, bpxnc([],[26 20],1000), perlc([], 1000), dtc([])};
names = {'loglc', 'fisherc', 'ldc', 'nmc', 'qdc', '1-NN', '2-NN', '3-NN', 'Parzen', 'bpxnc', 'perlc', 'dtc'};
evaluations_feat = evaluate_classifiers(nist_feat, classifiers, names);
%% Test profile dataset
classifiers = {loglc, fisherc, ldc, nmc, qdc, knnc([], 1), knnc([], 2), knnc([], 3), parzenc, bpxnc([],[32 20],1000), perlc([], 1000), dtc([])};
names = {'loglc', 'fisherc', 'ldc', 'nmc', 'qdc', '1-NN', '2-NN', '3-NN', 'Parzen', 'bpxnc', 'perlc', 'dtc'};
evaluations_prof = evaluate_classifiers(nist_profile, classifiers, names);

%% Test pixel dataset
classifiers = {loglc, fisherc, ldc, nmc, qdc, knnc([], 1), knnc([], 2), knnc([], 3), parzenc, bpxnc([],[32 20],1000), perlc([], 1000), dtc([])};
names = {'loglc', 'fisherc', 'ldc', 'nmc', 'qdc', '1-NN', '2-NN', '3-NN', 'Parzen', 'bpxnc', 'perlc', 'dtc'};
evaluations_pix = evaluate_classifiers(nist_pix, classifiers, names);
%% Test dissimilarity dataset - Euclidean distance
classifiers = {fisherc, ldc, nmc, qdc, knnc([], 1), knnc([], 2), knnc([], 3), parzenc, bpxnc([],[80 20],1000), perlc([], 1000), dtc([])};
names = {'fisherc', 'ldc', 'nmc', 'qdc', '1-NN', '2-NN', '3-NN', 'Parzen', 'bpxnc', 'perlc', 'dtc'};
evaluations_dis = evaluate_classifiers(nist_dis, classifiers, names);

%% Test dissimilarity dataset - Euclidean distance
classifiers = {fisherc, ldc, nmc, qdc, knnc([], 1), knnc([], 2), knnc([], 3), parzenc, bpxnc([],[80 20],1000), perlc([], 1000), dtc([])};
names = {'fisherc', 'ldc', 'nmc', 'qdc', '1-NN', '2-NN', '3-NN', 'Parzen', 'bpxnc', 'perlc', 'dtc'};
evaluations_dis_cos = evaluate_classifiers(nist_dis_cos, classifiers, names);
