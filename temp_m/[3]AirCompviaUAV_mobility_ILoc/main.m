clear;clc;
addpath('./afunction');
delete ('test/*.txt')
dirname1 = 'test4';
mkdir (dirname1);
RGBmatrix = [0.04, 0.09, 0.27;  %black 
             0.89,0.09,0.05; %red
              0.8500 0.3250 0.0980; %orange 
              1,0.6,0.07; % yellon 
              0.24,0.57,0.25;  %green
             0 0.4470 0.7410;]; %blue            
%% para
para.outerverb = 1;
para.innerverb = 1;
para.maxiter = 200;
para.flag = 0;
para.outertol = 1e-4;

T_set = 27;

K = 40; part = 2; num = floor(K*0.4); c_w = [100,100;250,200]; c_IF = [0,0]; 
c_r = 50; c_region = [0,0;500,300];
Vmax = 20; Vmin= 0; beta0 = 1e-4; alpha = 2;  H= 100;  delta = 0.2;
sigma = db2pow(-170+10*log10(10*1e6));
p1 = 5*db2pow(10)/10^3; p2 = 5*db2pow(10)/10^3;
p3 = p1; p4 = p1;
peak_P = [p1*ones(num,1);p2*ones(K-num,1)]; 
Aver_P = [p3*ones(num,1);p4*ones(K-num,1)]*0.5;
Ksize = num2cell([K,1,1,1,H,alpha,sigma,beta0,Vmax,Vmin,delta]);
Ksize{3} = peak_P;
Ksize{4} = Aver_P;
%% location
Loc.num = num; Loc.part = part; Loc.c_w = c_w; Loc.c_r = c_r; Loc.c_region = c_region;
[LocationX] = Location_user(Ksize,Loc,[],[]);
%%
save( [dirname1,'/data.mat'])


time1 = zeros(length(T_set),1); time2 = zeros(length(T_set),1);
for iT = 1:length(T_set)
    T = T_set(iT);
    N = T/delta ;
    Ksize{2} = N;
     [Location] = Location_user(Ksize,Loc,LocationX,'mobility');
    
    q_str = initial_straight_IF(N,c_IF);
    q_FHF = initial_ellipse(Ksize,c_IF,Location);
   
    dirname = num2str(iT);
    mkdir ([dirname1,'/',dirname]);
    
    para.scal0 = 1; para.scal1 = 1e2; para.scal2 = 1; para.scal3 = 1e2;
     rho1 = 1/sqrt(K); rho2 = 1/sqrt(K); rho3 = 4/sqrt(K);
     rho = [rho1,rho2,rho3];
     para.rho = rho;
    
    save([[dirname1,'/',dirname,'/'],['parameter']],'Ksize','dirname','iT','q0','N','para','Location');   
    
   
   tic
   [Variable2,allobj2,err2] = BCD_SCA_Block3(Ksize,Location,q_str,c_IF,para); 
   t2 = toc
   obj2 = allobj2(end);
   time2(iT) = t2;
   save([[dirname1,'/',dirname,'/'],['method2']],'Variable2','allobj2','err2','t2','obj2');
     
   tic 
   [Variable1,allobj1,err1,err_q1,err_obj1] = BCD_ADMM_Block3(Ksize,Location,q_str,c_IF,para);
   t1 = toc
   time1(iT) = t1;
   obj1 = allobj1(end);
   save([[dirname1,'/',dirname,'/'],['method1']],'Variable1','allobj1','err_q1','err1','err_obj1','t1','obj1');
   
    [Variable5,allobj5,err5] =  BCD_Block2_fixQ(Ksize,Location,q_FHF,c_IF,para);
   
   
   fprintf('iT: %.3d, MSE_ADMM:%.3e, MSE_SCA:%.3e\n',iT,obj1, obj2);  

end
save([dirname1,'/',['time']],'time1','time2');

flog1 =0;
if flog1
    close;  figure(1)
    vals = [time1, time2];
    b = bar(T_set,vals);
    b(2).FaceColor = [0 0.4470 0.7410];
    b(1).FaceColor = [0.8500 0.3250 0.0980];
    xlabel('Mission time $T (s)$','interpreter','latex','FontSize',16)
    ylabel('Simulation time (s)','interpreter','latex','FontSize',16)
    legend({'Proposed BCD-ADMM','BCD-SCA'},'interpreter','latex','FontSize',16);
    
    saveas(gcf,[dirname1,'/TimeT.fig']);
    print ([dirname1,'/TimeT.eps'], '-depsc2', '-r600')
%% Figure 2 
    close; figure(2)
    for iT = 1:2
         dirname2 = num2str(iT);
         load([[dirname1,'/',dirname2,'/'],['parameter']])  
         load([[dirname1,'/',dirname2,'/'],['method1']]) 
         q1 = Variable1.q; 
        plot([c_IF(1,1);q1(2:2:end,1);q1(N+1,1)],[c_IF(1,2);q1(2:2:end,2);q1(N+1,2)],'-^','LineWidth',2,'MarkerSize',6); hold on
    end
    
    plot(Location.userX,Location.userY,'*','Color',RGBmatrix(6,:),'MarkerSize',6); hold on
    h1 = plot(c_IF(1,1),c_IF(1,2),'<','Color',RGBmatrix(4,:),'MarkerSize',8); hold on
    set(h1,'MarkerFaceColor',get(h1,'color'));
    h2 = plot(c_IF(2,1),c_IF(2,2),'>','Color',RGBmatrix(5,:),'MarkerSize',8); hold on
    set(h2,'MarkerFaceColor',get(h2,'color'));
    grid on;
    set(gca,'GridLineStyle','--','GridColor','k', 'GridAlpha',0.2);
    xlim(c_region(:,1))
    h11 = legend('T = 25 $s$', 'T = 35 $s$');
    set(h11,'FontSize',16,'interpreter','latex');
    xlabel('$x$(m)','interpreter','latex','FontSize',16);
    ylabel('$y$(m)','interpreter','latex','FontSize',16);
    
    saveas(gcf,[dirname1,'/Traj.fig']);
    print ([dirname1,'/Traj.eps'], '-depsc2', '-r600'); close
  %% Figure 3  
     close; figure(3)
    for iT = 1:length(T_set)
        dirname2 = num2str(iT);
        load([[dirname1,'/',dirname2,'/'],['parameter']])
        load([[dirname1,'/',dirname2,'/'],['method1']])
        h1 = semilogy(allobj1(1:3:end),'-','LineWidth',2,'Color',RGBmatrix(2,:));hold on
        set(h1,'MarkerFaceColor',get(h1,'color'));
    end
    
    for iT = 1:length(T_set)
        dirname2 = num2str(iT);
        load([[dirname1,'/',dirname2,'/'],['parameter']])
        load([[dirname1,'/',dirname2,'/'],['method2']])
        h2 = semilogy(allobj2(1:3:end) ,'--','LineWidth',2,'Color',RGBmatrix(1,:));hold on
    end
    
    grid on;
    set(gca,'GridLineStyle','--','GridColor','k', 'GridAlpha',0.2);
   % ylim([1e-3 4e-3]);
    legend('Proposed BCD-ADMM','BCD-SCA','interpreter','latex','FontSize',14);
    xlabel('Iterations','interpreter','latex','FontSize',16);
    ylabel('Objective value','interpreter','latex','FontSize',16);
    
    saveas(gcf,[dirname1,'/Conver.fig']);
    print ([dirname1,'/Conver.eps'], '-depsc2', '-r600')
    close
    
  close; figure(3)
  center_mobility = Location.center_mobility;
  q1 = q_FHF;
    for i = 1:part
        plot(center_mobility(1:10:end,1,i),center_mobility(1:10:end,2,i),'--^','Color',RGBmatrix(i,:),'LineWidth',2,'MarkerSize',6); hold on
    end
    h1 = plot(c_IF(1,1),c_IF(1,2),'<','Color',RGBmatrix(4,:),'MarkerSize',8); hold on
    set(h1,'MarkerFaceColor',get(h1,'color'));
    h2 = plot(c_IF(2,1),c_IF(2,2),'>','Color',RGBmatrix(5,:),'MarkerSize',8); hold on
    set(h2,'MarkerFaceColor',get(h2,'color'));
    plot([c_IF(1,1);q1(2:2:end,1);q1(N+1,1)],[c_IF(1,2);q1(2:2:end,2);q1(N+1,2)],'-^','LineWidth',2,'MarkerSize',6); hold on
    grid on;
    set(gca,'GridLineStyle','--','GridColor','k', 'GridAlpha',0.2);  
    h11 = legend('T = 25 $s$', 'T = 35 $s$');
    set(h11,'FontSize',16,'interpreter','latex');
    xlabel('$x$(m)','interpreter','latex','FontSize',16);
    ylabel('$y$(m)','interpreter','latex','FontSize',16);  
    
    
end


