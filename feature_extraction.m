function [nist_feat_pca, nist_profile_pca, nist_pix_pca, nist_dis_pca] = feature_extraction(nist_feat, nist_profile, nist_pix, nist_dis,processed_data,percent);
    
    nist_feat_pca = nist_feat*pca(processed_data,percent);
    nist_profile_pca = nist_profile*pca(processed_data,percent);
    nist_pix_pca = nist_pix*pca(processed_data,percent);
    nist_dis_pca = nist_dis*pca(processed_data,percent);
    
end

