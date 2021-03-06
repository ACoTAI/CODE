% this program is the code of clique merging evolution (CLIME) algorithm to detect coalition from given network.
load('data\data.mat');% load network:adajcent matrix W, label Label
Max_Iter=5;% max iteration for each stage
l=10;% learning rate
exp_num_coalition=4;% expected number of coalition
result_all=[];% results of each k stages
Matrix_Clique=fun_extract_clique(W);% extraction of all clique
[Matrix_Clique, num_Clique_by_order,list_k]=fun_sort_Clique(Matrix_Clique);% sort all clique in order k from maximum order to minimum order, Matrix_Clique is the index matrix of all clique, num_Clique_by_order is number of cliques corresponding to oerder k, k_list is list of order k
index_skeleton_clique=0;% end index of order k-1 clique in Matrix_Clique
index_current_clique=0;% end index of order k clique in Matrix_Clique
theta=[];% distribution of coalitions
omega=rand(exp_num_coalition,1);% coalition mixing weight
omega=omega./sum(omega);
%% code of Algorithm 2: Clique Merging Evolution
for k=list_k % k stage
    num_added_clique=num_Clique_by_order(list_k==k);% number of order k clique
    index_current_clique=index_current_clique+num_added_clique;
    A=Matrix_Clique(:,1:index_current_clique);% index of all cliques in k stage
    Y=sparse(A'*A);% adjacent matrix of all cliques in k stage
    L=[l*ones(1,index_skeleton_clique) 1/l*ones(1,num_added_clique)];% make the cliques in k-1 stage with learning rate l, and make the cliques in k stage with learning rate 1/l 
    index_skeleton_clique=index_skeleton_clique+num_added_clique;
    theta(theta~=max(theta))=0;
    add_theta=rand(exp_num_coalition,num_added_clique);% initialize distribution of coalition for k-cliques
    add_theta=add_theta./sum(add_theta,2);
    theta=[theta,add_theta];% merge the distribution of coalitions in k-1 stage with the random initialization of k-cliques
    for loop=1:Max_Iter
        q=(L.*Y)*(omega.*theta)'; % derive q by omege and theta
        q=q./sum(q,2);
        theta=(Y*q)'./sum(q.*sum(Y,2))';% optimize theta with q fixed
        omega=sum(q,1)';% optimize omega with q fixed
        omega=omega./sum(omega);
    end
    result=A*q;% output
    result=result./sum(result,2);
    result_all=[result_all result];%record every coalition list of each stage
end
imagesc(result_all);% show all results
colorbar;
%description of Figure 1: it shows the result for all stages. The number in
%left is the index of nodes. For the columns from left to right, every
%exp_num_coalition number of columns combines one stage. 