%% Pre-process dataset

% Final image size after resize
image_size = 20;
% Takes every data_step entries for each class in the dataset
data_step = 4; 
processed_dataset = pre_process(data_step, image_size);

%% Construct the features datasets that are to be used
% nist_pix - Pixel datasets
% nist_dis - Euclidean dissimilarity dataset
% nist_dis_cos - Cosine dissimilarity dataset
[nist_pix, nist_dis, nist_dis_cos] = feature_generation(processed_dataset);

%% Choose classifiers
classifiers = {loglc, fisherc, knnc([], 1), knnc([], 3), parzenc, bpxnc([],[26 20],1000)};
names = {'Logistic', 'Fisher', '1-NN', '3-NN', 'Parzen', 'Neural Network'};

%% Perform PCA with optimum number for each classifier

%% Combine classifiers (optional)

%% Evaluate performance
errors_pix = struct(...
    'names', names, ...
    'classifiers' , classifiers);

errors = {3, 4, 1, 2, 3, 4};

[errors_pix.errors] = errors{:};

%%
std_pix = {};

errors_dis = struct(...
    'names', names(:, 2:length(names)), ...
    'classifiers' , classifiers(:, 2:length(classifiers)));
std_dis = {};

errors_dis_cos = struct(...
    'names', names(:, 2:length(names)), ...
    'classifiers' , classifiers(:, 2:length(classifiers)));
std_dis_cos = {};

[err, std] = prcrossval(nist_pix, classifiers, 10, 2);

%%
sorted_cell = sortCellArray(errors_pix, 3);

function sorted_cell = sortCellArray(arr, field_index)
    fields = fieldnames(arr);
    arrCell = struct2cell(arr);
    sz = size(arrCell); 
    
    arrCell = reshape(arrCell, sz(field_index), []);
    arrCell = arrCell';
    arrCell = sortrows(arrCell, field_index);
    
    arrCell = reshape(arrCell', sz);

    sorted_cell = cell2struct(arrCell, fields, 1);
end


