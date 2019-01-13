%% Tests various feature selection techniques and chooses the best one
function error_curves = feature_selection(num_feats, trn, tst, classf)
    % Individual selection
    w_i = featseli(trn, 'eucl-m', num_feats);
    e_i = clevalf(trn * w_i, classf, [1:1:num_feats], [], 2, tst * w_i);
    e_i.names = strcat(e_i.names, '-individual');
    
    % Backward selection
    w_b = featselb(trn, 'eucl-m', num_feats);
    e_b = clevalf(trn * w_b, classf, [1:1:num_feats], [], 2, tst * w_b);
    e_b.names = strcat(e_b.names, '-backward');
    
    % Forward selection
    w_f = featself(trn, 'eucl-m', num_feats);
    e_f = clevalf(trn * w_f, classf, [1:1:num_feats], [], 2, tst * w_f);
    e_f.names = strcat(e_f.names, '-forward');
    
    error_curves = {e_i, e_b, e_f};
end