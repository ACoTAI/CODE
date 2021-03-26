p=0.4;% edge generating probability of ER graph
q=0.01; %edge generating probability among ER graphs
p_overlap=0.3;% edge generating probability of overlap
structure_coaliton=[60 50 40];% size of each coalition
num_outlier=10;% num of outlier. If none, num_outlier=0
num_overlap=1;% num of overlap. If none, num_overlap=0
[W,Label]=fun_graph_generate(structure_coaliton,p,q,num_outlier,num_overlap,p_overlap);%W is the adjacent matrix of the generated network
save(['.\data\', 'data.mat'],'W','label');
