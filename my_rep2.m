function a = my_rep2(m)
    processed_dataset = pre_process_my_rep(m, 18);
%     simple_processed_dataset = simple_preprocess(m, 20);
    
%     [nist_feat, ~, ~, ~, ~] = feature_generation(simple_processed_dataset, true, false);
    [~, ~, nist_pix, ~, ~] = feature_generation(processed_dataset, false, false);
%     a = [nist_pix nist_pix];
    a = nist_pix;
end