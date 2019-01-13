function [nist_feat, nist_profile, nist_pix, nist_dis] = feature_generation(processed_dataset)
    nist_feat = im_features(processed_dataset, processed_dataset, 'all');
    nist_profile = im_profile(processed_dataset, 40, 40);
    nist_pix = processed_dataset;
    nist_dis = processed_dataset * proxm(processed_dataset);
end