function [Variable,allobj,err] = BCD_SCA_Block3(Ksize,Location,q0,c_IF,para)

[~,N,peak_P,Aver_P,~,~,~,~,~,~,~] = deal(Ksize{:});
maxiter = para.maxiter; 
verb = para.outerverb;
tol = para.outertol;
dname = para.dname;
fid1 =  fopen([dname,'/BCD_SCA_B3.txt'],'a+');
allobj = []; err = [];
lastobj = 1;
%% initialization
p_last = repmat(min(Aver_P',peak_P'),N,1);
Variable.p = p_last;
Variable.q  = q0;

for iter = 1:maxiter
   
  %% ######################given q,p, solve eta ####################
   [Denoise_eta,obj_eta,feasible_eta] = Denoise_subproblem(Ksize,Location,Variable,c_IF,para); 
  
    if  ~feasible_eta 
          fprintf(fid1,'iter:%.3d, feasible_eta:%.1d.\n',iter,feasible_eta);
        break;
    end
    allobj= [allobj;obj_eta];
   Variable.eta  = Denoise_eta;


   %% ######################given eta,q, solve p ####################
    [power_p,obj_p,feasible_p] = Power_subproblem_close(Ksize,Location,Variable,c_IF,para);     
    
    if  ~feasible_p 
        fprintf(fid1,'iter:%.3d, feasible_p:%.1d.\n',iter,feasible_p);
        break;
    end
    allobj= [allobj;obj_p]; 
    Variable.p = power_p;
  
  %% ######################given p,eta, solve q ####################     
    [Traj_q,obj_q,feasible_q] = Traj_subproblem_IF(Ksize,Location,Variable,c_IF,para);        
    
     
     if  ~feasible_q   
         fprintf(fid1,'iter:%.3d, feasible_q:%.1d.\n',iter,feasible_q);
        break;
    end
    allobj = [allobj;obj_q];
    Variable.q  = Traj_q; 
    
    err = [err; abs(lastobj - allobj(end))/lastobj]; 
    if verb && rem(iter,1) == 0
       fprintf(fid1,'iter:%.3d, obj_eta:%.3e, obj_p:%.3e, obj_q:%.3e, err:%.3e.\n',iter,obj_eta,obj_p,obj_q,err(end));
   
    end
    
  
    if  err(end)<tol 
        
        break;
    end
    
%% initialize the next iteration
    lastobj = allobj(end);
end

fclose(fid1);
end