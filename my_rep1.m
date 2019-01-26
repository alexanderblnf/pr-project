function a = my_rep1(m)
    processed_dataset = pre_process_my_rep(m, 20);
    simple_processed_dataset = simple_preprocess_eval(m, 20);
    
    [nist_feat, ~, ~, ~, ~] = feature_generation(simple_processed_dataset, true, false);
    [~, ~, nist_pix, ~, ~] = feature_generation(processed_dataset, false, false);
    a = [nist_pix nist_feat];
end