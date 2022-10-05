function [Block_matrix] = ADMM_matrix(Ksize,Variable,Location,c_IF)
[K,N,~,~,~,~,~,beta0,~,~,~] = deal(Ksize{:});

%%
if  isfield(Variable,'Theta')
    Theta = Variable.Theta ;
end

%% 
A1 = [eye(N)*-1,zeros(N,1)] + [zeros(N,1),eye(N,N)];
A2 = zeros(2,N+1);
A2(1,1) = 1; A2(2,N+1)=1; 
c = c_IF;
Dn = [zeros(N,1),eye(N,N)]; 
%%
w = Location.w; 
B2 = (mat2cell(w,N*ones(1,K),2));
B2 = cellfun(@(x) [c_IF(1,:) ;x], B2,'UniformOutput',false);

Block_matrix.A1 = sparse(A1);
Block_matrix.A2 = sparse(A2);
Block_matrix.B2 = B2;
Block_matrix.c = c;
Block_matrix.Dn = sparse(Dn);

temp_Theta = [zeros(1,K);Theta]./beta0;
temp_B1 = (num2cell(sqrt(temp_Theta),1))';
B1 = cellfun(@(x) diag(x), temp_B1,'UniformOutput',false);

temp_B = sum(temp_Theta,2);
B = diag(temp_B);




B1B2 = cellfun(@(x,y) x*y, B1,B2,'UniformOutput',false);
B1B1B2 = cellfun(@(x,y) x'*y,B1,B1B2,'UniformOutput',false);

Block_matrix.B1 = B1;
Block_matrix.B = B;
Block_matrix.B1B2 = B1B2;
Block_matrix.B1B1B2 = B1B1B2;
end