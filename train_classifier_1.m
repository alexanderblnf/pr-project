%% Generate datasets
image_size = 20;
dataset_step = 4;
[processed_dataset_2500, data] = pre_process(dataset_step, image_size);
processed_dataset_simple_2500 = simple_preprocess(dataset_step, image_size);

[nist_feat, ~, ~, ~, ~] = feature_generation(processed_dataset_simple_2500, true, false, false);
[~, ~, nist_pix, ~, ~] = feature_generation(processed_dataset_2500, false, false, false);

%% Train classifier

dataset1 = nist_pix;
dataset2 = nist_feat;

W1 = dataset1 * pcam([], 29) * parzenc;
W2 = loglc;

parallel = [W1; W2];

comb = parallel * minc;

w1 = [dataset1 dataset2] * comb;
