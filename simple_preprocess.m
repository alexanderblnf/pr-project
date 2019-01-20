function processed_dataset = simple_preprocess(a, image_size)

% a = prnist([0:9],[1:dataset_step:1000]);

%% Pre-process dataset (make all images square and bring them to same size)
processed_dataset = im_box(a,0,1);
processed_dataset = im_resize(processed_dataset, [image_size, image_size]);
processed_dataset = im_box(processed_dataset, 1, 0);
processed_dataset = prdataset(processed_dataset);
end