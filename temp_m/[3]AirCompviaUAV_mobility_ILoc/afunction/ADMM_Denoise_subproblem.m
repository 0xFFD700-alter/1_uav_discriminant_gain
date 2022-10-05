function [Denoise_eta,oriobj,feasible_eta] = ADMM_Denoise_subproblem(Ksize,Location,Variable,c_IF,para)

[~,N,~,~,~,~,sigma,~,~,~,~] = deal(Ksize{:});

if  isfield(Variable,'q')
    Traj_q = Variable.q ;    
end

if  isfield(Variable,'Theta')
    Theta = Variable.Theta ;
end

verb = para.innerverb;
temp_sqr = sum(Theta,2);
temp = sum(sqrt(Theta),2);
Denoise_eta = ((sigma+temp_sqr)./(temp)).^2;
Variable.eta  = Denoise_eta;


[oriobj,~,~] = Obj_fun_f(Ksize,Location,Variable);
[ feasible_eta] = CheckFunction(Ksize,Location,Variable,c_IF,verb);
end

