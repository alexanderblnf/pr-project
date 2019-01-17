%% Generate datasets
image_size = 20;
dataset_step = 2;
processed_dataset = pre_process(dataset_step, image_size);
[nist_feat, nist_profile, nist_pix, nist_dis] = feature_generation(processed_dataset);

%% Select classifiers
classifiers = {loglc, fisherc, knnc([], 1), knnc([], 3), parzenc, bpxnc([],[26 20],1000)};
names = {'loglc', 'fisherc', '1-NN', '3-NN', 'Parzen', 'bpxnc'};

%% Test feature dataset
evaluations_feat = evaluate_classifiers(nist_feat, classifiers, names);

%% Test profile dataset
evaluations_prof = evaluate_classifiers(nist_profile, classifiers, names);

%% Test pixel dataset
evaluations_pix = evaluate_classifiers(nist_pix, classifiers, names);

%% Test dissimilarity dataset
classifiers = {fisherc, knnc([], 1), knnc([], 3), parzenc, bpxnc([],[26 20],1000)};
names = {'fisherc', '1-NN', '3-NN', 'Parzen', 'bpxnc'};

evaluations_dis = evaluate_classifiers(nist_dis, classifiers, names);
