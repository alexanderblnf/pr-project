function [combinedP, combinedS, names] = combine_2_classifiers_1_pca(data1, num_pc, c1, c2)

parallel = [];
seq = [];

w1 = data1 * pcam([], num_pc);

W1 = w1 * c1;
W2 = c2;

parallel = [W1; W2];
seq = [W1 W2];

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


