function [W,Label]=fun_graph_generate(structure_coaliton,p,q,num_outlier,num_overlap,p_overlap)
W=[];%adjicent matrix
Label=[];%label
num_coalition=length(structure_coaliton);
for i=1:num_coalition
    W=blkdiag(W,rand(structure_coaliton(i)));%join coalitons
    Label=[Label; i*ones(structure_coaliton(i),1)];
end
W(W>(1-p))=1;
W(W<=(1-p))=0;
W_q=rand(size(W));%construct edges between coalitions
W_q=W_q.*(1-logical(W));
W_q(W_q>(1-q))=1;
W_q(W_q<=(1-q))=0;
W=W+W_q;
W=W-diag(diag(W));
W=triu(W);
W=W+W';
pred=0;
while pred~=num_outlier
    adj_outlier=rand(size(W,1),1);
    adj_outlier(adj_outlier<1-q)=0;
    adj_outlier(adj_outlier>=q)=1;
    W=[W adj_outlier];
    adj_outlier=[adj_outlier;0];
    W=[W;adj_outlier'];
    Label=[Label;i+1];
    pred=pred+1;
end
pred=0;
while pred~=num_overlap
    adj_overlap=rand(size(W,1),1);
    adj_overlap(adj_overlap<1-p_overlap)=0;
    adj_overlap(adj_overlap>=p_overlap)=1;
    W=[W adj_overlap];
    adj_overlap=[adj_overlap;0];
    W=[W;adj_overlap'];
    Label=[Label;i+2];
    pred=pred+1;
end
end