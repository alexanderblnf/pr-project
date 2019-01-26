%% Step 1 - Evaluate classifier without image pre-processing
initial_classifier_analysis; 

%% Step 2 - Evaluate classifier with image pre-processing
preprocess_initial_classifier_analysis;

%% Step 3.1 - PCA Analysis - Run for each dataset
dataset = nist_feat;
classifiers = {ldc, fisherc, knnc([], 1), parzenc, nmc};
names = {'loglc', 'fisherc', '1-NN', 'Parzen', 'nmc'};

results = feature_extraction_frac(dataset, classifiers);

%% Step 3.2 - Display image for each dataset
for i = 1 : length(results)
    plot(0.1: 0.1 : 1, results{i}, 'LineWidth', 1.5);
    hold on;
end
legend(names);
xlabel('Percentage of retained variance');
ylabel('Error');
title('PCA Analysis');

%% Step 3.2 - Find the number of principal components and evaluate classifier
start = 0.75;
step = 0.01;
stop = 0.85;

dataset = nist_pix;
classifier = parzenc;

for i = start : step : stop
   fprintf('---- i: %.2f\n', i);
   [err, stdd, n] = feature_extraction_2(dataset, classifier, i);
   fprintf('err: %.3f\t std: %.3f\t n: %d\n', err, stdd, n);
end

%% Step 4 - Feature selection analysis
feature_selection_analysis;

%% Step 4.1 - Plot the errors
plote(errC_pix_all);
figure;
plote(errC_feat_all);
figure;
plote(errC_prof_all);
%% Step 5 - Combine classifiers - First combination
c1 = parzenc;
c2 = knnc([], 1);

dataset1 = nist_pix;
dataset2 = nist_pix;

num_pc = [29, 30];

[combinedP, combinedS, names] = combine_2_classifiers_pca(dataset1, dataset2, num_pc, c1, c2);

%% Evaluate parallel combination
[errP, stdP] = prcrossval([dataset1 dataset2], combinedP, 10, 1);

%% Evaluate stack combination
[errS, stdS] = prcrossval(dataset1, combinedS, 10, 2);

%% Second combination
dataset1 = nist_pix;
dataset2 = nist_feat;

c1 = parzenc;
c2 = loglc;

num_pc = 29;

[combinedP, combinedS, names] = combine_2_classifiers_1_pca(dataset1, num_pc, c1, c2);

%% Evaluate parallel combination
[errP, stdP] = prcrossval([dataset1 dataset2], combinedP, 10, 1);

%% Combine 3 classifiers - First combination
c1 = parzenc;
c2 = knnc([], 1);
c3 = knnc([], 3);

dataset = nist_pix;

num_pc = [29, 30, 27];

[combinedP, combinedS, names] = combine_3_classifiers_pca(dataset, num_pc, c1, c2, c3);

%% Evaluate parallel combination
[errP, stdP] = prcrossval([dataset dataset dataset], combinedP, 10, 1);

%% Evaluate stack combination
[errS, stdS] = prcrossval(dataset1, combinedS, 10, 2);

%% Combine 3 classifiers - Second combination
c1 = parzenc;
c2 = loglc;
c3 = parzenc;

dataset1 = nist_pix;
dataset2 = nist_feat;
dataset3 = nist_profile;

num_pc = 29;

[combinedP, combinedS, names] = combine_3_classifiers_1_pca(dataset1, num_pc, c1, c2, c3);

%% Evaluate parallel combination
[errP, stdP] = prcrossval([dataset1 dataset2 dataset3], combinedP, 10, 1);

%% Step 6 - Evaluate system - Train selected classifier

dataset1 = nist_pix;
dataset2 = nist_feat;

W1 = dataset1 * pcam([], 29) * parzenc;
W2 = loglc;

parallel = [W1; W2];

comb = parallel * minc;

w1 = [dataset1 dataset2] * comb;

%% Evaluate
errors_s1 = zeros(1, 5);
for i = 1 : 5
    errors_s1(i) = nist_eval('my_rep1', w1, 50);
end
disp(min(errors_s1));
disp(mean(errors_s1));
disp(max(errors_s1));