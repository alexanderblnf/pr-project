function [nist_feat, nist_pix, nist_dis, nist_dis_cos] = feature_generation(processed_dataset)
    nist_feat = im_features(processed_dataset, processed_dataset, 'all');
    nist_pix = processed_dataset;
    nist_dis = processed_dataset * proxm(processed_dataset);
    nist_dis_cos = processed_dataset * proxm(processed_dataset, 'o');
end