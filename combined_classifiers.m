function [combinedP, combinedS, names] = combined_classifiers(data, classifiers)

n = length(classifiers);
parallel = [];
seq = [];

for i = 1 : n
    w = data * classifiers{:, i};
    parallel = [parallel; w];
    seq = [seq w];
end

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


