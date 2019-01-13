function processed_dataset = pre_process(size)
    a = prnist([0:9],[1:4:1000]);
    
    %% Pre-process dataset (make all images square and bring them to same size)
    processed_dataset = im_box(a,0,1);
    processed_dataset = im_resize(processed_dataset, [size, size]);
    processed_dataset = im_box(processed_dataset, 1, 0);
    processed_dataset = prdataset(processed_dataset);
end