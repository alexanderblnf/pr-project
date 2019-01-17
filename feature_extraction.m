function [nist_feat_pca, nist_profile_pca, nist_pix_pca, nist_dis_pca] = feature_extraction(processed_dataset,percent);
    
    nist_feat_pca = im_features(processed_dataset, 'all')*pca([],percent);
    nist_profile_pca = im_profile(processed_dataset, 40, 40)*pca([],percent);
    nist_pix_pca = processed_dataset*pca([],percent);
    nist_dis_pca = processed_dataset*proxm(processed_dataset)*pca([],percent);
    
end

