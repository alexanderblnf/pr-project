function [combinedP, combinedS, names] = combine_3_classifiers_pca(data, num_pc, c1, c2, c3)

w1 = data * pcam([], num_pc(1));
w2 = data * pcam([], num_pc(2));
w3 = data * pcam([], num_pc(3));

W1 = w1 * c1;
W2 = w2 * c2;
W3 = w3 * c3;

parallel = [W1; W2; W3];
seq = [W1 W2 W3];

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


