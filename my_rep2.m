function a = my_rep2(m)
    processed_dataset = pre_process_my_rep(m, 18);    
    [~, ~, nist_pix, ~, ~] = feature_generation(processed_dataset, false, false, false);
    a = nist_pix;
end