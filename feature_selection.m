%% Tests various feature selection techniques and chooses the best one
function [error_curves, optimal_feats] = feature_selection(num_feats, trn, tst, classf, step)
    % Individual selection
    [w_i, r_i] = featseli(trn, 'eucl-m', num_feats);
    e_i = clevalf(trn * w_i, classf, [1:step:num_feats], [], 2, tst * w_i);
    e_i.names = strcat(e_i.names, '-individual');
    
    % Backward selection
    [w_b, r_b] = featselb(trn, 'eucl-m', num_feats);
    e_b = clevalf(trn * w_b, classf, [1:step:num_feats], [], 2, tst * w_b);
    e_b.names = strcat(e_b.names, '-backward');
    
    % Forward selection
    [w_f, r_f] = featself(trn, 'eucl-m', num_feats);
    e_f = clevalf(trn * w_f, classf, [1:step:num_feats], [], 2, tst * w_f);
    e_f.names = strcat(e_f.names, '-forward');
    
    error_curves = {e_i, e_b, e_f};
    feature_mapping = {r_i(:, 3)', w_b{:, 1}, r_f(:, 3)'};
    optimal_feats = get_optimal_feats(error_curves, feature_mapping);
end

function optimal_feats = get_optimal_feats(error_curves, feature_mapping)
    optimal_feats = {};
    for i = 1 : length(error_curves)
        [error, index] = min(error_curves{:, i}.error);
        mapping = feature_mapping{:, i};
        s = struct;
        s.error = error;
        s.index = index;
        s.mapping = mapping(:, 1:index);
        s.names = error_curves{:, i}.names;
        optimal_feats{end + 1} = s;
    end
end