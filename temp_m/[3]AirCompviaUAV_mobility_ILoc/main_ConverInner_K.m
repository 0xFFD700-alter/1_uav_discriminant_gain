clear;clc;
addpath('./afunction');
mkdir ('test');

%% para
para.outerverb = 1;
para.innerverb = 1;
para.maxiter = 200;
para.flag = 0;

K_set = 40; part = 2;  c_w = [100,200;300,250]; c_IF = [0,0;400,300];
Vmax = 20; Vmin= 0; beta0 = 1e-4; alpha = 2;  H= 100;  delta = 0.2;
sigma = db2pow(-170+10*log10(10*1e6));

T = 30;
N = T/delta ;
Ksize = num2cell([1,N,1,1,H,alpha,sigma,beta0,Vmax,Vmin,delta]);

%% location
q0 = initial_straight_IF(N,c_IF);
%%
save test/data.mat

for iK = 1:length(K_set)
    K = K_set(iK);
    num = floor(K*0.4);
    peak_P = [db2pow(10)/10^3*2*ones(num,1);db2pow(10)/10^3*ones(K-num,1)]; 
    Aver_P = peak_P*0.7;
    Ksize{1} = K;
    Ksize{3} = peak_P;
    Ksize{4} = Aver_P;
    Loc.K = K; Loc.num = num; Loc.part = part; Loc.c_w = c_w;
    [Location,c_geom,r_cp] = Location_user(Loc);
    dirname = num2str(iK);
    para.scal2 = 1/(K*N*5);
    rho1 = 1/sqrt(2*N); rho2 = 1/sqrt(2*N); rho3 = 6/sqrt(2*N); 
    rho = [rho1,rho2,rho3];
    para.rho = rho;   
    mkdir ('test/',dirname);
    save([['test/',dirname,'/'],['parameter']],'Ksize','dirname','iK','K','Location','para');   
    
   %load([['test/',dirname,'/'],['parameter']])  
   %load([['test/',dirname,'/'],['method2']]) 
   
   [Variable1,allobj1,err1,err_q1,err_obj1] = BCD_ADMM_Block3(Ksize,Location,q0,c_IF,para);
   save([['test/',dirname,'/'],['method1']],'Variable1','allobj1','err_q1','err1','err_obj1');
   
    
   
   
   [Variable2,allobj2,err2] = BCD_SCA_Block3(Ksize,Location,q0,c_IF,para); 
   save([['test/',dirname,'/'],['method2']],'Variable2','allobj2','err2');
   
   fprintf('iK: %.3d, MSE_ADMM:%.3e, MSE_SCA:%.3e\n',iK,allobj1(end),allobj2(end));  

end




RGBmatrix = [0.04, 0.09, 0.27;  %black            
             0 0.4470 0.7410; %blue
             0.8500 0.3250 0.0980; %orange 
             0.89,0.09,0.05; %red
             1,0.6,0.07; % yellon          
             0.24,0.57,0.25;];  %green
save([['test/'],['RGBmatrix']],'RGBmatrix');  

flog1 =0;
if flog1
    figure(1)
    for iT = 1:length(T_set)
         dirname = num2str(iT);
         load([['test/',dirname,'/'],['parameter']])  
         load([['test/',dirname,'/'],['method2']]) 
        q2 = Variable2.q; 
       plot([c_IF(1,1);q2(2:2:end,1);q2(N+1,1)],[c_IF(1,2);q2(2:2:end,2);q2(N+1,2)],'-^','LineWidth',2,'MarkerSize',6); hold on
    end
    
    plot(Location.userX,Location.userY,'*','Color',RGBmatrix(3,:),'MarkerSize',6); hold on
    h1 = plot(c_IF(1,1),c_IF(1,2),'<','Color',RGBmatrix(4,:),'MarkerSize',8); hold on
    set(h1,'MarkerFaceColor',get(h1,'color'));
    h2 = plot(c_IF(2,1),c_IF(2,2),'>','Color',RGBmatrix(4,:),'MarkerSize',8); hold on
    set(h2,'MarkerFaceColor',get(h2,'color'));
    grid on;
    set(gca,'GridLineStyle','--','GridColor','k', 'GridAlpha',0.2);  
    h11 = legend('Proposed trajectory','Users locations');
    set(h11,'FontSize',11);
    xlabel('$x$(m)','interpreter','latex','FontSize',14);
    ylabel('$y$(m)','interpreter','latex','FontSize',14);
    
    saveas(gcf,'test/Traj.fig');
    print Traj.eps -depsc2 -r600
 
    figure(2)
       
    h1 = plot(allobj3(1:3:end),'--','LineWidth',2,'MarkerSize',2);hold on 
    set(h1,'MarkerFaceColor',get(h1,'color'));
    h2 = plot(allobj4(1:3:end) ,'-','LineWidth',2,'MarkerSize',2);hold on 
    set(h2,'MarkerFaceColor',get(h2,'color'));
    h3 = plot(allobj5(1:3:end) ,'-.','LineWidth',2,'MarkerSize',2);hold on 
    set(h3,'MarkerFaceColor',get(h3,'color')); 
    grid on;
    set(gca,'GridLineStyle','--','GridColor','k', 'GridAlpha',0.2);
    ylim([0.5e-4 4e-3]);
    h11 = legend('SCA','Proposed-CVX','Proposed-ADMM');
    set(h11,'FontSize',11);
    xlabel('Iterations','interpreter','latex','FontSize',14);
    ylabel('Average MSE','interpreter','latex','FontSize',14);
    
    saveas(gcf,'test/Iter.fig');
    print Iter.eps -depsc2 -r600
    
    
    figure(2)
       
    h1 = plot(allobj3q(1:3:end),'--','LineWidth',2,'MarkerSize',2);hold on 
    set(h1,'MarkerFaceColor',get(h1,'color'));
    h2 = plot(allobj4q(1:3:end) ,'-','LineWidth',2,'MarkerSize',2);hold on 
    set(h2,'MarkerFaceColor',get(h2,'color'));
    h3 = plot(allobj5(1:3:end) ,'-.','LineWidth',2,'MarkerSize',2);hold on 
    set(h3,'MarkerFaceColor',get(h3,'color')); 
    grid on;
    set(gca,'GridLineStyle','--','GridColor','k', 'GridAlpha',0.2);
    ylim([0.5e-4 4e-3]);
    h11 = legend('SCA','Proposed-CVX','Proposed-ADMM');
    set(h11,'FontSize',11);
    xlabel('Iterations','interpreter','latex','FontSize',14);
    ylabel('Average MSE','interpreter','latex','FontSize',14);
    
    saveas(gcf,'test/Iter1.fig');
    print Iter1.eps -depsc2 -r600

end


