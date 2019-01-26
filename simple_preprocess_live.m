function processed_dataset_feat = simple_preprocess_live(dataset, image_size)
  
    % Pre-process dataset (make all images square and bring them to same size)
    a = data2im(dataset);
    dt = {};
    [~, ~, ~, n] = size(a);

    for i = 1 : n
        ds = im_box(a(:, :, :, i),0,1);
        ds = im_resize(ds, [image_size, image_size]);
        ds = im_box(ds, 1, 0);
        dt{i} = im_features(ds, ds, 'all');
        lab{i}= strcat('digit_',num2str(getlabels(dataset(i,:))));
    end

    processed_dataset_feat = prdataset(dt,lab);
end