function [power_p,oriobj,feasible_p] = Power_subproblem(Ksize,Location,Variable,c_IF,para)

if  isfield(Variable,'eta')
    Denoise_eta = Variable.eta ;
end
if  isfield(Variable,'q')
    Traj_q = Variable.q ;
end
verb = para.innerverb;
%%
[K,N,peak_P,Aver_P,~,~,~,~,~,~,~] = deal(Ksize{:});
q_time = Traj_q(2:N+1,:);
[beta_pl] = Channel_LargeFading(Ksize,Location,q_time);
eta_rep = repmat(Denoise_eta,1,K);
Cons_mat = sqrt(beta_pl./eta_rep);
power_p = zeros(N,K);
for ik = 1:K
    Cons_k = Cons_mat(:,ik);
    %% optimization
    cvx_begin quiet
   % cvx_solver mosek
    variable p(N,1) nonnegative    
    %% objective function   
    minimize sum_square_abs(p.*Cons_k-1)
    %% constraint
    subject to  
       sum_square_abs(p) <= Aver_P(ik)*N;
       p.^2 <= peak_P(ik);
    cvx_end
    
    power_p(:,ik) = p.^2;
end
Variable.p = power_p;
[oriobj,~] = Obj_fun(Ksize,Location,power_p,Denoise_eta,Traj_q);
[feasible_p] = CheckFunction(Ksize,Location,Variable,c_IF,verb);
end

