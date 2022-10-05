function [Variable,allobj,err,err_q,err_obj] = BCD_ADMM_Block3(Ksize,Location,q0,c_IF,para)

[K,N,peak_P,Aver_P,~,~,~,beta0,~,~,~] = deal(Ksize{:});
dname = para.dname;
fid1 =  fopen([dname,'/BCD_ADMM_B3.txt'],'a+');
maxiter = para.maxiter; 
verb = para.outerverb;
tol = para.outertol;
scal0 =  para.scal0;
allobj = []; lastobj = 1; err = [];
%% initialization
p_last = repmat(min(Aver_P',peak_P'),N,1);
q_time = q0(2:N+1,:);
[beta_pl] = Channel_LargeFading(Ksize,Location,q_time);
Theta = p_last.*beta_pl;
Variable.Theta = Theta;
Variable.q  = q0;
q = q0;

err_q = cell(maxiter,1);

for iter = 1:maxiter
   
  %% ######################given q,theta, solve eta ####################
   [Denoise_eta,obj_eta,feasible_eta] = ADMM_Denoise_subproblem(Ksize,Location,Variable,c_IF,para);
    
    if  ~feasible_eta 
        fprintf(fid1,'iter:%.3d, feasible_eta:%.1d.\n',iter,feasible_eta);
        break;
    end
     allobj= [allobj;obj_eta];
    Variable.eta  = Denoise_eta;
    
    %% ######################given eta,q, solve theta ####################
    [Theta,obj_Theta,feasible_Theta] = ADMM_Theta_subproblem(Ksize,Location,Variable,c_IF,para);  
    
     if  ~feasible_Theta 
        fprintf(fid1,'iter:%.3d, feasible_Theta:%.1d.\n',iter,feasible_Theta); 
        break;
     end
     
     
     allobj = [allobj;obj_Theta]; 
     Variable.Theta = Theta;
   %% ######################given p,eta, solve q ####################     
   
    [q,obj_q,feasible_q, err_q{iter},err_obj{iter},q_star] = ADMM_Traj_subproblem_new2(Ksize,Location,Variable,c_IF,para);  
    if  ~feasible_q
        fprintf(fid1,'iter:%.3d, feasible_q:%.1d.\n',iter,feasible_q);
        break;
    end
    Variable.q  = q;     
    allobj = [allobj;obj_q]; 
 
     
    err = [err; abs(lastobj - allobj(end))/lastobj]; 
    if verb 
        fprintf(fid1,'iter:%.3d,obj_eta:%.3e, obj_Theta:%.3e, obj_q:%.3e, err:%.3e.\n',iter,obj_eta,obj_Theta,obj_q,err(end));       
    end  
    if  err(end)<tol || iter == maxiter 
        [beta_pl] = Channel_LargeFading(Ksize,Location,q(2:N+1,:));
        p = Theta./beta_pl;
        Variable.p  = p;
        break;
    end

    lastobj = allobj(end);
  
end

fclose(fid1);
end