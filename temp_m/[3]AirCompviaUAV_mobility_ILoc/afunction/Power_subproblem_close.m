function [power_p,oriobj,feasible_p] = Power_subproblem_close(Ksize,Location,Variable,c_IF,para)

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
   
    a = Cons_mat(:,ik);
    P1k = peak_P(ik);
    P2k = Aver_P(ik);
    if  sum_square_abs(min([a./(a.^2), sqrt(P1k)*ones(N,1)],[],2)) - P2k*N <= 1e-12
        u = min([a./(a.^2), sqrt(P1k)*ones(N,1)],[],2);       
    else
      x = 0; y = 1;
      f1 = sum_square_abs(min([a./(a.^2 + x), sqrt(P1k)*ones(N,1)],[],2))-N*P2k;
      f2  = sum_square_abs(min([a./(a.^2 + y), sqrt(P1k)*ones(N,1)],[],2))-N*P2k;
      while f2 > 0
            y = y*10;
            f2  = sum_square_abs(min([a./(a.^2 + y), sqrt(P1k)*ones(N,1)],[],2))-N*P2k;
      end
      
      
      while abs(f1) > 1e-12
          
          xx = (x+y)/2;
          f1 = sum_square_abs(min([a./(a.^2 + xx), sqrt(P1k)*ones(N,1)],[],2))-N*P2k;
          
          if f1 > 0
              x = xx;
          elseif f1 < 0
              y = xx;
          end
          
      end  
       u = min([a./(a.^2 + xx), sqrt(P1k)*ones(N,1)],[],2);
    end 
     
    power_p(:,ik) = (u.^2);
    
end
Variable.p = power_p;
[oriobj,~,~] = Obj_fun_f(Ksize,Location,Variable);
[feasible_p] = CheckFunction(Ksize,Location,Variable,c_IF,verb);
end

