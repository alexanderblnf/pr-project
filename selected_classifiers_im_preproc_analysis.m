%% Generate datasets
image_size = 20;
dataset_step = 4;
[processed_dataset, data] = pre_process(dataset_step, image_size);
%%
dataset_step = 4;
[processed_dataset_simple] = simple_preprocess(dataset_step, image_size);
[nist_feat_big_simple, nist_profile_big_simple, ~, ~, ~] = feature_generation(processed_dataset_simple, true, true, false);

%%
[nist_feat_big, ~, nist_pix_big, nist_dis_big, nist_dis_cos_big] = feature_generation(processed_dataset, true, false, true);

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
dataset2 = nist_feat;
dataset3 = nist_profile_big_simple;
c1 = parzenc;
c2 = ldc;
c3 = ldc;
num_pc = [32, 16];
[combinedP, combinedS, names] = combined_classifiers(dataset1, num_pc,c1, c2, c3);
%%
[errP, stdP] = prcrossval([dataset1 dataset1 dataset2], combinedP, 10, 1);
%%
[errS, stdS] = prcrossval(dataset1, combinedS, 10, 2);

%%
dataset = nist_feat;
    classifier = bpxnc([], [30 20], 1000);
[M, N] = size(dataset);

result = feature_extraction_2(dataset, classifier, 2, N);

%%
dataset = nist_pix;
classifier = bpxnc([30 10], 1000);

W = dataset * pcam([], 43);
[err_N, std_N] = prcrossval(dataset, W * classifier, 10, 5);

%%

evaluations = evaluations_prof;
for i = 1 : length(evaluations)
    disp(evaluations{:, i});
end

%% Train best classifier
dataset1 = nist_pix;
dataset2 = nist_dis;

classifier = parzenc;

W1 = dataset1 * pcam([], 32) * parzenc;
W2 = dataset1 * pcam([], 42) * knnc([], 1);

parallel = [W1; W2];
seq = [W1 W2];
comb = parallel * prodc;
comb1 = seq * prodc;

% w = W * classifier; 
w = dataset1 * comb1;
%%
dataset1 = nist_pix;
dataset2 = nist_feat;
% dataset3 = nist_dis;

W1 = dataset1 * pcam([], 30) * parzenc;
W2 = loglc;
% W3 = dataset3 * pcam([], 32) * ldc;

parallel = [W1; W2];

comb = parallel * prodc;

w = [dataset1 dataset2] * comb;
%%
dataset1 = nist_pix;
dataset2 = nist_feat;

W1 = dataset1 * pcam([], 30) * parzenc;
W2 = dataset1 * pcam([], 16) * ldc;
% W3 = dataset3 * pcam([], 32) * ldsc;

seq = [W1 W2];

comb = seq * minc;

w = dataset1 * comb;

%%
dataset1 = nist_pix_big;
dataset2 = nist_feat_big_simple;
dataset3 = nist_profile_big_simple;

c1 = parzenc;
c2 = loglc;
c3 = parzenc;

W1 = dataset1 * pcam([], 29) * c1;
W2 = c2;
% W3 = dataset3 * pcam([], 32) * ldc;

parallel = [W1; W2];

comb = parallel * prodc;

w = [dataset1 dataset2] * comb;
%%
dataset1 = nist_pix_big;
comb = (perlc*qdc([],[],1e-6));
W = dataset1 * comb;
[err1, st1d1] = prcrossval(dataset, comb, 10, 5);

%%
errors_s2_1 = zeros(1, 10);
for i = 1 : 10
    errors_s2_1(i) = nist_eval('my_rep', w, 10);
end
disp(min(errors_s2_1));
disp(mean(errors_s2_1));
disp(max(errors_s2_1));

%%
classifiers = {ldc, fisherc, knnc([], 1), parzenc, nmc};
names = {'ldc', 'fisherc', '1-NN', 'parzenc', 'nmc'};
results_prof = feature_extraction_frac(nist_dis_cos, classifiers);

%%
for i = 1 : length(results_prof)
    plot(0.1: 0.1 : 1, results_prof{i}, 'LineWidth', 1.5);
    hold on;
end
legend(names);
xlabel('Percentage of retained variance');
ylabel('Error');
title('PCA Analysis - Dissimilarity Cosine');