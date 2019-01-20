function [combinedP, combinedS, names] = combined_classifiers(data1, data2, num_pc, classifier1, classifier2)

parallel = [];
seq = [];

% for i = 1 : n
%     w = data * classifiers{:, i};
%     parallel = [parallel; w];
%     seq = [seq w];
% end
w1 = pcam(data1, num_pc(1));
% w2 = pcam(data2, num_pc(2));
% w3 = pcam(data, num_pc(3));


W1 = w1 * classifier1;
% W2 = w2 * classifier2;
% W3 = w3 * classifier3;

% parallel = [W1; W2];
% seq = [W1 W2];

parallel = [W1; classifier2];
seq = [W1 classifier2];

p_mean = parallel * meanc;
p_min = parallel * minc;
p_max = parallel * maxc;
p_prod = parallel * prodc;
p_median = parallel * medianc;

combinedP = {p_mean, p_min, p_max, p_prod, p_median};

s_mean = seq * meanc;
s_min = seq * minc;
s_max = seq * maxc;
s_prod = seq * prodc;
s_median = seq * medianc;

combinedS = {s_mean, s_min, s_max, s_prod, s_median};

names = {'mean', 'min', 'max', 'prod', 'median'};


