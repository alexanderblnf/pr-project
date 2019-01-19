function [nist_feat, nist_profile, nist_pix, nist_dis, nist_dis_cos] = feature_generation(processed_dataset, include_im_feats, include_im_profile)
    nist_feat = [];
    nist_profile = [];

    if include_im_feats == true
        nist_feat = im_features(processed_dataset, processed_dataset, 'all');
    end
    
    if include_im_profile == true
        nist_profile = im_profile(processed_dataset, 22, 22);
    end
    
    nist_pix = processed_dataset;
    nist_dis = processed_dataset * proxm(processed_dataset);
    nist_dis_cos = processed_dataset * proxm(processed_dataset, 'o');
end