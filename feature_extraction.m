function [nist_feat_pca, nist_pix_pca, nist_dis_pca] = feature_extraction(processed_dataset,percent);
    
    nist_feat_pca = im_features(processed_dataset, 'all')*pcam([],percent);
    nist_pix_pca = processed_dataset*pcam([],percent);
    nist_dis_pca = processed_dataset*proxm(processed_dataset)*pcam([],percent);
    
end

