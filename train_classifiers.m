% train a number of classifiers on the input training set
% outputs the mapping resulted from the training
function W = train_classifiers(trn)

all = {loglc; fisherc; ldc; nmc; qdc; knnc([], 1); knnc([], 2); parzenc;
    treec; svc; bpxnc;};

W = trn * all;