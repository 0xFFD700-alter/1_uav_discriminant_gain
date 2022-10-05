function [A_P,A_theta,S_P,S_theta] = Obj_fun_power(Ksize,Loc,Location,Variable)
part = Loc.part; num = Loc.num;
[K,N,~,~,~,~,~,~,~,~,delta] = deal(Ksize{:});
if  isfield(Variable,'q')
    q = Variable.q ;  
    q_time = q(2:N+1,:);
    [beta_pl] = Channel_LargeFading(Ksize,Location,q_time);
end


if  isfield(Variable,'p')
    p = Variable.p ;   
    Theta = p.*beta_pl;
elseif  isfield(Variable,'Theta')
    Theta = Variable.Theta ;
    p = Theta./beta_pl;
end

A_P = zeros(N,part); A_theta = zeros(N,part);

for iM = 1: part
        if iM == part
            K_index = (iM-1)*num+1: K;           
        else
            K_index = (iM-1)*num+1: (iM)*num;
        end
        A_P(:,iM) = sum(p(:,K_index),2)/length(K_index);
        A_theta(:,iM) = sum(Theta(:,K_index),2)/length(K_index);
end

S_P = sum(p,2);
S_theta = sum(Theta,2);

end

