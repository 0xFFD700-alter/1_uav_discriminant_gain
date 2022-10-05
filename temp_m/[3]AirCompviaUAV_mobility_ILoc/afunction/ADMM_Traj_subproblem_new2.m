function [new_q,oriobj,feasible_q, err_q, err_obj, q_star] = ADMM_Traj_subproblem_new2(Ksize,Location,Variable,c_IF,para)

dname = para.dname;
verb = para.innerverb;
flag = para.flag;
rho = para.rho;
scal0 =  para.scal0; scal1 =  para.scal1; scal2 =  para.scal2; scal3 =  para.scal3;

fid1 =  fopen([dname,'/ADMM_q.txt'],'a+'); 
maxiter = 1e4;
[K,N,peak_P,Aver_P,H,~,~,beta0,Vmax,~,delta] = deal(Ksize{:});

if  isfield(Variable,'q')
    Traj_q = Variable.q ;   
end
if  isfield(Variable,'Theta')
    Theta = Variable.Theta ;
end



%% Initial
q = Traj_q;
Lambda(1:K,1) = {zeros(N+1,2)};
Xi(1:K,1) = {zeros(N+1,2)};
Tau = zeros(N,2); 
%% Matrix
rho1 = rho(1); rho2 = rho(2); rho3 = rho(3); 
%% Matrix
w = Location.w; w_cell = (mat2cell(w,N*ones(1,K),2));
hat_P = beta0*repmat((peak_P)', N,1 )./Theta-H^2;
sum_Theta = sum(Theta,1);
tilde_P = ((Aver_P)'*N - sum_Theta./beta0*H^2);
%% Matrix
[Block_matrix] = ADMM_matrix(Ksize,Variable,Location,c_IF);
A1 = Block_matrix.A1; A2 = Block_matrix.A2;
B1 = Block_matrix.B1; B2 = Block_matrix.B2;
c =  Block_matrix.c;  Dn =  Block_matrix.Dn;
B = Block_matrix.B; B1B2 = Block_matrix.B1B2; B1B1B2 = Block_matrix.B1B1B2;
temp_a =  ones(K,1);
%temp_a =  tilde_P';
temp_Bf =  cellfun(@(x,y) x'*x./y,B1,(num2cell(temp_a)),'UniformOutput',false);  
Bf = sum(cat(3,temp_Bf{:}),3);  
scal2 = max(B(:))-min(B(:)); scal0 = 1/(max(Bf(:))-min(Bf(:))*10);
AA = rho1*K*eye(size(B,1),size(B,1))/scal1 + rho2*B/scal2 + 2*Bf*scal0 + rho3*(A1'*A1/scal3);
inver_AA = pinv(AA);

%% ground truth
err_q = []; err_obj = []; q_star = 0;
if flag
      [q_star,~,~,~] = ADMM_TrajZ_subproblem_CVX(Ksize,Location,Variable,c_IF,para);
      err_q = [err_q;norm(Traj_q-q_star,'fro')/norm(q_star,'fro')];
      cvx_obj = sum(cellfun(@(x,y,t)norm(x*q_star-x*y,'fro')^2/t,B1,B2,(num2cell(temp_a))));
      A_obj = sum(cellfun(@(x,y,t)norm(x*q-x*y,'fro')^2/t,B1,B2,(num2cell(temp_a))));
      err_obj = [err_obj;norm(A_obj-cvx_obj)/cvx_obj];
      fprintf(fid1,'cvx_obj:%.3e, obj:%.3e, relative_errq:%.3e, relative_errobj:%.3e.\n',cvx_obj, A_obj, err_q(end),err_obj(end));
end


%% iteration
for iter = 1:maxiter
     
    %% Gamma update      
       temp1 = cellfun(@(x,y) Dn*q/sqrt(scal1) - Dn*x - y/sqrt(scal1), Lambda, w_cell, 'UniformOutput', false);
       abs_temp1 = cellfun(@(x) sqrt(sum(x.^2,2)), temp1, 'UniformOutput', false);
       abs_temp2 = cell2mat(abs_temp1');
       min_coeff = bsxfun(@min, sqrt(hat_P)/sqrt(scal1)./abs_temp2,1 );
       new_Gamma = cellfun(@(x,y,t) [c(1,:); (x*ones(1,2)).*y + t/sqrt(scal1)],(num2cell(min_coeff,1))',temp1, w_cell, 'UniformOutput', false);
       
       %% V update         
       temp1 = cellfun(@(x,y,t) x*q/sqrt(scal2) - y - t/sqrt(scal2), B1, Xi, B1B2, 'UniformOutput', false);
       abs_temp1 = cellfun(@(x) norm(x,'fro'), temp1);
       min_coeff = bsxfun(@min, sqrt(tilde_P')/sqrt(scal2)./abs_temp1,1);
       new_V = cellfun(@(x,y,t) x*y + t/sqrt(scal2), num2cell(min_coeff), temp1, B1B2, 'UniformOutput', false);
     
    %% z update      
        Aq = A1 * q/sqrt(scal3) - Tau;      
        abs_temp1 = sqrt(sum(Aq.^2,2));
        min_coeff = bsxfun(@min, Vmax*delta/sqrt(scal3)./abs_temp1,1);
        new_z = (min_coeff*ones(1,2)).*Aq;      
    %% q update       
        
        temp_part1 =  cellfun(@(x,y) (x+y)/sqrt(scal1), new_Gamma,Lambda,'UniformOutput',false);     
        part1 = rho1*sum(cat(3,temp_part1{:}),3); 
               
        temp_part2 =  cellfun(@(x,y,t) x'*(y+t)/sqrt(scal2),B1,new_V,Xi,'UniformOutput',false);     
        part2 = rho2*sum(cat(3,temp_part2{:}),3);    
               
        part3 = rho3*A1'/sqrt(scal3)*(new_z + Tau); 
        
        temp_part4 =  cellfun(@(x,y,t) x./y,B1B1B2,(num2cell(temp_a)),'UniformOutput',false);  
        part4 = sum(cat(3,temp_part4{:}),3)*2*scal0;      
             
        part = part1 + part2 + part3 + part4;
              
        temp_q = inver_AA*part; 
        new_q = temp_q;
        new_q(1,:) =  c(1,:);
       
       
    %% Lagrange multipliers update       
     new_Lambda  = cellfun(@(x,y)x + y - new_q/sqrt(scal1), Lambda,new_Gamma,'UniformOutput',false);     
     new_Xi  = cellfun(@(x,y,t) x + y - t*new_q/sqrt(scal2), Xi,new_V,B1,'UniformOutput',false);         
     new_Tau = Tau +  new_z - (new_q(2:end,:) - new_q(1:N,:))/sqrt(scal3);
    
  %%  residual  
     obj_lambda  = sum(cellfun(@(x,y)norm(x-y,'fro')^2,new_Lambda,Lambda));   
     obj_xi  = sum(cellfun(@(x,y)norm(x-y,'fro')^2,new_Xi,Xi));    
     obj_tau = norm(new_Tau - Tau,'fro')^2;    
     obj_q = norm(new_q - q,'fro')^2;
   
  %%   ADMM obj
    if verb && rem(iter,500) == 0
     A_obj = sum(cellfun(@(x,y,t)norm(x*q-x*y,'fro')^2/t,B1,B2,(num2cell(temp_a))));
     fprintf(fid1,'i:%.3d, lambda:%.2e, xi:%.2e, tau:%.2e, q:%.2e, obj:%.2e.\n', iter, obj_lambda,obj_xi,obj_tau,obj_q,A_obj);
    end
    q = new_q ;
    Lambda = new_Lambda;
    Xi = new_Xi;
    Tau = new_Tau;
    if flag
        err_q = [err_q;norm(q-q_star,'fro')/norm(q_star,'fro')];
        A_obj = sum(cellfun(@(x,y,t)norm(x*q-x*y,'fro')^2/t,B1,B2,(num2cell(temp_a))));
        err_obj = [err_obj;norm(A_obj-cvx_obj)/cvx_obj];
        if verb && rem(iter,1) == 0
        fprintf(fid1,'i:%.3d, obj:%.3e, relative errq:%.2e, relative errobj:%.2e.\n',iter,A_obj,err_q(end),err_obj(end));  
        end
    end
    %% terminal condition 
     if obj_lambda < 1e-4 && obj_xi < 1e-4 && obj_tau < 1e-4 && obj_q < 1e-4
            fprintf(fid1,'i:%.3d, lambda:%.2e, xi:%.2e, tau:%.2e, q:%.2e.\n', iter, obj_lambda,obj_xi,obj_tau,obj_q);
        break;
     end  
    
end

Variable.q = new_q;
[feasible_q] = CheckFunction(Ksize,Location,Variable,c_IF,verb);
[oriobj,~,~] = Obj_fun_f(Ksize,Location,Variable);

fclose(fid1);
end
