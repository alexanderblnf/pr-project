%% Generate dataset from picture
livewriting;

%% Process data
image_size = 20;

live_processed_dataset_feat = simple_preprocess_live(live_dataset, 20);
[live_processed_dataset, ~] = pre_process_live(live_dataset, 20);

%% Get train dataset
dataset_step = 4;
[processed_dataset, ~] = pre_process(dataset_step, image_size);
processed_dataset_simple = simple_preprocess(dataset_step, image_size);

[~, ~,nist_pix, ~, ~] = feature_generation(processed_dataset, true, false);
[nist_feat, ~, ~, ~, ~] = feature_generation(processed_dataset_simple, true, false);

%% Train classifier
dataset1 = nist_pix;
dataset2 = nist_feat;

W1 = dataset1 * pcam([], 29) * parzenc;
W2 = loglc;

parallel = [W1; W2];

comb = parallel * prodc;

w = [dataset1 dataset2] * comb;

%% 
errLive = [live_processed_dataset live_processed_dataset_feat] * w * testc;