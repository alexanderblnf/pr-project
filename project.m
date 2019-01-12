a = prnist([0:9],[1:4:1000]);

%% Pre-process dataset (make all images square and bring them to same size)
processed_dataset = im_box(a,0,1);
processed_dataset = im_resize(processed_dataset, [40, 40]);
processed_dataset = prdataset(processed_dataset);

%% Features
features = im_features(processed_dataset, processed_dataset, 'all');


%% Train classifier features - featsel 
[trn, tst] = gendat(features, 0.5);
[m, n] = size(features);
w = featself(trn, 'maha-s', 20);
eS = clevalf(trn * w, fisherc, [1:1:20], [], 5, tst * w);

% plote(e);

%% Train classifier features - feat extraction
[trn, tst] = gendat(features, 0.5);
w = scalem(trn, 'variance') * pcam(trn, 20);
eE = clevalf(trn * w, fisherc, [1:1:20], [], 5, tst * w);

%%
plote({eS, eE});


%% Features - variant 2
profile = im_profile(processed_dataset, 16, 16);

%% Train classifier profile - featsel 
[trn, tst] = gendat(profile, 0.2);
[m, n] = size(profile);
w = featself(trn, 'eucl-m', 32);
eS = clevalf(trn * w, svc(proxm('e')), [1:2:32], [], 1, tst * w);

% plote(e);

%% Train classifier profile - feat extraction
[trn, tst] = gendat(profile, 0.2);
wE = scalem(trn, 'variance') * pcam(trn, 20);
eE = clevalf(trn * w, svc(proxm('e')), [1:2:32], [], 1, tst * w);

%%
plote({eS, eE});

%%
[err, std] = prcrossval(profile, w * svc(proxm('e')), 10, 2);