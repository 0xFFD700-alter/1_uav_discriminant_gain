function [obj,f1,f2] = Obj_fun_f(Ksize,Location,Variable)

[K,N,~,~,H,alpha,sigma,beta0,~,~,~] = deal(Ksize{:});

if  isfield(Variable,'q')
    Traj_q = Variable.q ;    
end

if  isfield(Variable,'Theta')
    Theta = Variable.Theta ;
    
elseif  isfield(Variable,'p')
    power_p = Variable.p ;
    q_time = Traj_q(2:N+1,:);
    [beta_pl] = Channel_LargeFading(Ksize,Location,q_time);
    Theta = power_p.* beta_pl;
end

if  isfield(Variable,'eta')
    Denoise_eta = Variable.eta ;
end

Cons_matrix = sqrt(Theta)./sqrt(repmat(Denoise_eta,1,K)); 
f1_r = (Cons_matrix-1).^2;
f2_r = sigma./Denoise_eta;
f1 = sum(f1_r(:))/(K^2*N);
f2 = sum(f2_r(:))/(K^2*N);
obj = f1+f2;
end

