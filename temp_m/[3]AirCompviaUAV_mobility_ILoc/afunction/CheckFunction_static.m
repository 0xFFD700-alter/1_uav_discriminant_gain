function [ feasible] = CheckFunction_static(Ksize,Location,Variable)


[K,N,peak_P,Aver_P,H,alpha,sigma,beta0,Vmax,Vmin,delta] = deal(Ksize{:});

if  isfield(Variable,'q')
    q = Variable.q ;  
    q_time = q(2:N+1,:);
    [beta_pl] = Channel_LargeFading(Ksize,Location,q_time);
end

if  isfield(Variable,'Theta')
    Theta = Variable.Theta ;
    
elseif  isfield(Variable,'p')
    p = Variable.p ;   
    Theta = p.*beta_pl;
end

if  isfield(Variable,'eta')
    eta = Variable.eta ;
end

feasible1 = 1;
for ik = 1:K
     tempflag = (Theta(:,ik)./beta_pl(:,ik) <= peak_P(ik)+ peak_P(ik)*1e-2);
     flag1 = (all(tempflag)==1);
     flag2 = (sum(Theta(:,ik)./beta_pl(:,ik))/N <= Aver_P(ik) + Aver_P(ik)*1e-2);
     if ~(flag1 && flag2)
         fprintf('ik:%.3d, flag1:%.1d, flag2:%.1d, power constraint cannot be met!\n',ik,flag1,flag2);
         fprintf('ik:%.3d, P2:%.3e.\n',ik,sum(Theta(:,ik)./beta_pl(:,ik))/N);
         feasible1 = 0;
         break;
     end
end

feasible = feasible1 ;
end