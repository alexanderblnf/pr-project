%% Generate datasets
image_size = 18;
dataset_step = 100;
[small_processed_dataset, data] = pre_process(dataset_step, image_size);

[~, ~, nist_pix, ~, ~] = feature_generation(small_processed_dataset, false, false, false);

%% Train classifier
dataset1 = nist_pix;

W2 = dataset1 * pcam([], 25) * parzenc;

w2 = dataset1 * W2;