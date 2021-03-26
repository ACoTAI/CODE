function [ Matrix_Clique_sort, num_Clique_by_order,list_k] = fun_sort_Clique( M )
%This funciton is to sort all cliques from max-clique to min-clique
order_M=sum(M,1);% calculate the order of each clique
list_k=sort(unique(order_M),'descend');
Matrix_Clique_sort=[];
%Matrix_Clique_weighted=[];
num_Clique_by_order=[];

for k=list_k
    Matrix_Clique_sort=[Matrix_Clique_sort,M(:,order_M==k)];%store clique matrix by the order k
    %Matrix_Clique_weighted=[Matrix_Clique_weighted,k*M(:,order_M==k)];%weight index vector of clique by order k
    num_Clique_by_order=[num_Clique_by_order, sum(sum(M(:,order_M==k))~=0)];
end
%%show distribution of all clique
% image_allclique_distribution=figure;
% imagesc(Matrix_Clique_weighted);
% colorbar;
% name_all_clique=('Distribution of All Cliques');
% set(image_allclique_distribution,'Name',name_all_clique);
% hold off;
end



