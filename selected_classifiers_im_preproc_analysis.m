%% Generate datasets
image_size = 20;
dataset_step = 4;
% [processed_dataset] = simple_preprocess(dataset_step, image_size);
[processed_dataset, data] = pre_process(dataset_step, image_size);
%%
[nist_feat, ~, nist_pix, nist_dis, nist_dis_cos] = feature_generation(processed_dataset, true, false);

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
dataset2 = nist_dis;
dataset1 = nist_pix;
num_pc = 29;
classifier2 = fisherc;
classifier1 = parzenc;
classifier3 = fisherc;
num_pc1 = 194;
num_pc2 = 10;
% [combinedP, combinedS, names] = combined_classifiers(dataset, classifier1, num_pc1, classifier2, num_pc2);
[combinedP, combinedS, names] = combined_classifiers(dataset1, dataset2, num_pc, classifier1, classifier2);
%%
[errP, stdP] = prcrossval([dataset1 dataset2], combinedP, 10, 1);
%%
[errS, stdS] = prcrossval([dataset1;dataset2], combinedS, 10, 1);

%%
dataset = nist_dis;
classifier = fisherc;

result = feature_extraction(dataset, classifier, true);

%%
dataset = nist_dis_cos;
classifier = {loglc, fisherc, knnc([], 1), knnc([], 3), parzenc};
results_dis_cos = feature_extraction_frac(dataset, classifier);

%%
names = {'loglc', 'fisherc', '1-NN', '3-NN', 'parzenc'};
for i = 1 : length(results_dis_cos)
    plot(0.1: 0.1 : 1, results_dis_cos{i}, 'LineWidth', 1.5);
    hold on;
end
legend(names);
xlabel('Percentage of retained variance');
ylabel('Error');
title('PCA Analysis - Dissimilarity Cosine');
%%
dataset = nist_pix;
classifier = fisherc;
[w, n] = dataset * pcam([], 137);
[err, std] = prcrossval(dataset, w * classifier, 10, 5);
disp(err);
disp(n);