function [nist_feat_pca, nist_profile_pca, nist_pix_pca, nist_dis_pca] = feature_extraction(processed_data_set,percent);
    
    nist_feat_pca = im_features(processed_dataset, processed_dataset, 'all')*pca(processed_data_set,percent);
    nist_profile_pca = im_profile(processed_dataset, 40, 40)*pca(processed_data_set,percent);
    nist_pix_pca = processed_dataset*pca(processed_dataset,percent);
    nist_dis_pca = processed_dataset*proxm(processed_dataset)*pca(processed_data_set,percent);
    
end

