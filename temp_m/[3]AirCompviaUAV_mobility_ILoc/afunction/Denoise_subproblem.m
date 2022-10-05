function [Denoise_eta,oriobj,feasible_eta] = Denoise_subproblem(Ksize,Location,Variable,c_IF,para)

if  isfield(Variable,'p')
    power_p = Variable.p ;
end
if  isfield(Variable,'q')
    Traj_q = Variable.q ;   
end
verb = para.innerverb;
[~,N,~,~,~,~,sigma,~,~,~,~] = deal(Ksize{:});
q_time = Traj_q(2:N+1,:);
[beta_pl] = Channel_LargeFading(Ksize,Location,q_time);
Cons_matrix = power_p.* beta_pl;
temp_sqr = sum(Cons_matrix,2);
temp = sum(sqrt(Cons_matrix),2);
Denoise_eta = ((sigma+temp_sqr)./(temp)).^2;

Variable.eta = Denoise_eta;
[oriobj,~,~] = Obj_fun_f(Ksize,Location,Variable);
[feasible_eta] = CheckFunction(Ksize,Location,Variable,c_IF,verb);

end

