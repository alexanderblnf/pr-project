%% Step 1 - Evaluate classifier without image pre-processing
initial_classifier_analysis_2; 

%% Step 2 - Evaluate classifier with image pre-processing
preprocess_initial_classifier_analysis_2;

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
feature_selection_analysis_2;

%% Step 4.1 - Plot the errors
plote(errC_pix_all);
figure;
plote(errC_feat_all);
figure;
plote(errC_prof_all);
%% Step 5 - Combine classifiers - First combination
c1 = parzenc;
c2 = knnc([], 1);
% c2 = ldc;

dataset1 = nist_pix;
dataset2 = nist_pix;

num_pc = [25, 25];
% num_pc = [25, 31];

[combinedP, combinedS, names] = combine_2_classifiers_pca(dataset1, dataset2, num_pc, c1, c2);

%% Evaluate parallel combination
[errP, stdP] = prcrossval([dataset1 dataset2], combinedP, 10, 1);

%% Evaluate stack combination
[errS, stdS] = prcrossval(dataset1, combinedS, 10, 2);

%% Second combination
dataset1 = nist_pix;
dataset2 = nist_feat;
% dataset2 = dataset2 * featseli(dataset2, 'eucl-m', 18);

c1 = parzenc;
c2 = ldc;

num_pc = 25;

[combinedP, combinedS, names] = combine_2_classifiers_1_pca(dataset1, num_pc, c1, c2);

%% Evaluate parallel combination
[errP, stdP] = prcrossval([dataset1 dataset2], combinedP, 10, 1);

%% Step 6 - Evaluate system - Train selected classifier

dataset1 = nist_pix;

W1 = dataset1 * pcam([], 25) * parzenc;

combS = W1 * meanc;

w1 = dataset1 * combS;

%% Evaluate
errors_s2 = zeros(1, 10);
for i = 1 : 20
    errors_s2(i) = nist_eval('my_rep2', w1, 10);
end
disp(min(errors_s2));
disp(mean(errors_s2));
disp(max(errors_s2));