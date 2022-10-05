function [ feasible] = CheckFunction(Ksize,Location,Variable,c_IF,verb)


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
   
    a1=num2str(peak_P(ik));
    nn1 = length(a1)-find(a1=='.');
    a2 =  num2str(Aver_P(ik));
    nn2 = length(a2)-find(a2=='.');
    
    tempflag = ((Theta(:,ik)./beta_pl(:,ik)) <= peak_P(ik) + 1*10^(-nn1));
    flag1 = (all(tempflag)==1);
    flag2 = (sum(Theta(:,ik)./beta_pl(:,ik))/N <= Aver_P(ik)+ 1*10^(-nn2));
     if ~(flag1 && flag2)
         if verb
         in = find(tempflag~=1);
         fprintf('ik:%.3d, flag1:%.1d, flag2:%.1d, power constraint cannot be met!\n',ik,flag1,flag2);
         fprintf('ik:%.3d,P1:%.3e, P2:%.3e.\n',ik,(Theta(in,ik)./beta_pl(in,ik)),sum(Theta(:,ik)./beta_pl(:,ik))/N);
         end
         feasible1 = 0;
         break;
     end
end
feasible2 = 1;
tempflag3 = (sum((abs(q(2:end,:) - q(1:N,:))).^2,2) <= (Vmax*delta)^2 + (Vmax*delta)^2*1e-2);
flag3 = all(tempflag3==1);

tempflag4 = ( q(1,:) == c_IF(1,:));
flag4 = all(tempflag4==1);




if ~(flag3 && flag4 )
    if verb
    fprintf('flag3:%.1d, flag4:%.1d, Trajectory constraint cannot be met!\n',flag3,flag4);    
    end
    feasible2 = 0;
end
feasible = (feasible1 && feasible2);
end