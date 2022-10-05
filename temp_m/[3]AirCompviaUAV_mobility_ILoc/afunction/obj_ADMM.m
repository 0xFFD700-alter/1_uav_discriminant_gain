function [obj,obj4,vec_obj] = obj_ADMM(Ksize,Block_matrix,q,V,Gamma,z,Lambda,Xi,Tau,scal,rho)
[~,N,~,~,~,~,~,beta0,~,~,~] = deal(Ksize{:});
rho1 = rho(1); rho2 = rho(2); rho3 = rho(3); 
A1 = Block_matrix.A1;
B1 = Block_matrix.B1;
B2 = Block_matrix.B2;
Dn =  Block_matrix.Dn;


obj1 = rho1/2*sum(cellfun(@(x,y)norm(x + y - q,'fro')^2,Lambda,Gamma));
obj2 = rho2/2*sum(cellfun(@(x,y,t)norm(x + y - t*q,'fro')^2,Xi,V,B1));
obj3 = rho3/2*norm(A1*q/sqrt(scal) -z + Tau,'fro')^2;
obj4 = sum(cellfun(@(x,y)norm(x*q-x*y,'fro')^2,B1,B2));

obj = obj1 + obj2 + obj3 + obj4;
vec_obj = [obj1,obj2,obj3,obj4];
end