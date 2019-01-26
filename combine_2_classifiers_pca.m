function [combinedP, combinedS, names] = combine_2_classifiers_pca(data1, data2, num_pc, c1, c2)

parallel = [];
seq = [];

w1 = data1 * pcam([], num_pc(1));

if isempty(data2) == true
   w2 = data1 * pcam([], num_pc(2));
else
   w2 = data2 * pcam([], num_pc(2));
end

W1 = w1 * c1;
W2 = w2 * c2;

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


