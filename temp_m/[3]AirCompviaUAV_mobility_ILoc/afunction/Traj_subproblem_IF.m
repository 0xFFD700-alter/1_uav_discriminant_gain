function [Traj_q,oriobj,feasible_q] = Traj_subproblem_IF(Ksize,Location,Variable,c_IF,para)

dname = para.dname;
fid1 =  fopen([dname,'/TrajSubproblem.txt'],'a+'); 
if  isfield(Variable,'p')
    power_p = Variable.p ;
end
if  isfield(Variable,'eta')
    Denoise_eta = Variable.eta ;
end
if  isfield(Variable,'q')
    Traj_q = Variable.q ;
    last_q = Traj_q ;
end
scal = para.SCAscal; verb = para.innerverb;
[K,N,~,~,H,alpha,sigma,beta0,Vmax,Vmin,delta] = deal(Ksize{:});
q_time = last_q(2:N+1,:);
w_B3 = Location.w;


Cons_matrix = power_p*beta0./repmat(Denoise_eta,1,K); 
a_cons = vec(Cons_matrix);

X = repmat(eye(N),K,1); X = sparse(X);
q_rep = X*q_time;

d_2D = sum(abs(q_rep-w_B3).^2,2);
d_3D = d_2D + H^2;
f1 = a_cons./((d_3D).^(alpha/2));
f2 = 2*sqrt(a_cons)./((d_3D).^(alpha/4));
f3 = ones(N*K,1);
f4 = sigma./Denoise_eta;

f2_1 = 2*sqrt(a_cons)./((d_3D).^(alpha/4));
f2_2 = -alpha*sqrt(a_cons)./(2*(d_3D).^(alpha/4+1));
V1 = 2*(q_rep-w_B3);
C2 = f2_1 - f2_2 .*d_2D;
    
%% optimization
    cvx_begin quiet
    cvx_solver mosek
    variable Traj_q(N+1,2)
    variable s(K,N) nonnegative    
    %% objective function
    expression  d2D(N*K,1)   
    expression  f1(N*K,1) 
    expression  f2(N*K,1)  
    d2D = sum(square_abs((X*Traj_q(2:N+1,:)-w_B3)),2);
    d3D = H^2 + s(:);
    f1 = a_cons.*pow_p(d3D,-alpha/2);
    f2 = C2 + f2_2.*d2D;
    minimize 1/(K^2)*(sum(f1) - sum(f2))
    %% constraint
    subject to  
    %% Constraint 1   
    s(:)/scal <= (d_2D + sum(V1* (Traj_q(2:N+1,:) - q_time)'.*X,2) )/scal  
    %% Constraint 3   
    sum(square_abs(Traj_q(2:end,:) - Traj_q(1:N,:)),2) <= (Vmax*delta)^2;   
    %% Constraint 4    
     Traj_q(1,:) == c_IF(1,:); 
     
    
    cvx_end
    %% the problem
    if strcmp(cvx_status,'Infeasible') || strcmp(cvx_status,'Failed')
        %% optimization
        cvx_begin quiet
        cvx_solver sedumi
        variable Traj_q(N+1,2)
        variable s(K,N) nonnegative
        %% objective function
        expression  d2D(N*K,1)
        expression  f1(N*K,1)
        expression  f2(N*K,1)
        d2D = sum(square_abs((X*Traj_q(2:N+1,:)-w_B3)),2);
        d3D = H^2 + s(:);
        f1 = a_cons.*pow_p(d3D,-alpha/2);
        f2 = C2 + f2_2.*d2D;
        minimize 1/(K^2)*(sum(f1) - sum(f2))
        %% constraint
        subject to
        %% Constraint 1
        s(:)/scal <= (d_2D + sum(V1* (Traj_q(2:N+1,:) - q_time)'.*X,2) )/scal
        %% Constraint 3
        sum(square_abs(Traj_q(2:end,:) - Traj_q(1:N,:)),2) <= (Vmax*delta)^2;
        %% Constraint 4
        Traj_q(1,:) == c_IF(1,:);
               
        cvx_end
        if strcmp(cvx_status,'Infeasible') || strcmp(cvx_status,'Failed')
            Traj_q = nan;
            oriobj = nan;
            feasible_q = 0;
            display(cvx_status);
            fprintf(fid1,'Trajectory_subproblem: It is infeasible;\n');
            return;
        end
    end
    Traj_q(1,:) = c_IF(1,:); 
   
    Variable.q =  Traj_q;
    [feasible_q] = CheckFunction(Ksize,Location,Variable,c_IF,verb);
    [oriobj,~,~] = Obj_fun_f(Ksize,Location,Variable);
    fprintf(fid1,'cvx_obj:%.3e,ori_obj:%.3e.\n',cvx_optval,oriobj);
    
fclose(fid1);
end

