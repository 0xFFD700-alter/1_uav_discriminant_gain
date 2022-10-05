clear;clc;
addpath('./afunction');
dname = ['outerT_txt'];
mkdir (dname);
delete ([dname,'/*.txt'])
dirname1 = 'test_outerT';
mkdir (dirname1);

%% para
Generate_parameter
para.dname = dname; 

T_set = 20:20:100; 
len = length(T_set);
[obj1,obj2] = deal(zeros(len,1));

%% location
LocationX = Location_user(Ksize,Loc,[],'no');
save( [dirname1,'/data.mat'])

for ix = 1:len
    N = T_set(ix)/delta ;
    P2_set = P_set.*[N/2,N/2];
    p3 = P2_set(1,1); p4 = P2_set(1,2);
    Aver_P = [p3*ones(num,1);p4*ones(K-num,1)]/N;
    Ksize{4} = Aver_P;
    Ksize{2} = N;
    dirname2 = num2str(ix);
    dirname = [dirname1,'/',dirname2];
    mkdir (dirname);      
    Location = Location_user(Ksize,Loc,LocationX,'mobility');
    q_strFloc = initial_straight_lastLoc(Ksize,c_IF,Location);
    
    save([dirname,'/',['parameter']],'Ksize','dirname','ix','N','Location','q_strFloc');      
end  



ik = 1;
para1 = para;
para1.scal0 = 1; para1.scal1 = 1; para1.scal2 = 1; para1.scal3 = 1;
rho1 = 1; rho2 = 1; rho3 = 10;
rho = [rho1,rho2,rho3];
para1.rho = rho;
for ix = 1:len
    dirname2 = num2str(ix);
    dirname = [dirname1,'/',dirname2];
    load([dirname,'/',['parameter']])
    [Variable1,allobj1,err1,~,~] = BCD_ADMM_Block3(Ksize,Location,q_strFloc,c_IF,para1);
     temp = allobj1;
     obj1(ix, 1) = temp(end);
    save([dirname,'/',['method',num2str(ik)]],['Variable',num2str(ik)],['allobj',num2str(ik)],['err',num2str(ik)]);
    save([dirname1,'/',['method',num2str(ik)]],['obj',num2str(ik)]);
end
    
    
     
    

ik = 2;
for ix = 1:len
    dirname2 = num2str(ix);
    dirname = [dirname1,'/',dirname2];
    load([dirname,'/',['parameter']])
    [Variable2,allobj2,err2] = BCD_SCA_Block3(Ksize,Location,q_strFloc,c_IF,para);
    temp = allobj2;
    obj2(ix, 1) = temp(end);
    save([dirname,'/',['method',num2str(ik)]],['Variable',num2str(ik)],['allobj',num2str(ik)],['err',num2str(ik)]);
    save([dirname1,'/',['method',num2str(ik)]],['obj',num2str(ik)]);
end






 

flog1 =0;
if flog1
    close; figure(1)
    for ix = 1:length(T_set)
         dirname2 = num2str(ix);
         load([[dirname1,'/',dirname2,'/'],['parameter']])  
         load([[dirname1,'/',dirname2,'/'],['method2']]) 
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
    
 
    saveas(gcf,[dirname1,'/Traj.fig']);
    print ([dirname1,'/Traj.eps'], '-depsc2', '-r600')

    
   
    close; figure(2)
    for ix = 2:length(T_set)
        dirname2 = num2str(ix);
        load([[dirname1,'/',dirname2,'/'],['parameter']])
        load([[dirname1,'/',dirname2,'/'],['method1']])
        h1 = semilogy(err1,'-','LineWidth',2,'Color',RGBmatrix(ix,:));hold on
        set(h1,'MarkerFaceColor',get(h1,'color'));
    end
    
    for ix = 2:length(T_set)
        dirname2 = num2str(ix);
        load([[dirname1,'/',dirname2,'/'],['parameter']])
        load([[dirname1,'/',dirname2,'/'],['method2']])
        h2 = semilogy(err2 ,'--','LineWidth',2,'Color',RGBmatrix(ix,:));hold on
    end
    
    grid on;
    set(gca,'GridLineStyle','--','GridColor','k', 'GridAlpha',0.2);
   % ylim([0.5e-4 4e-3]);
    h11 = legend('T = 35 $s$','T = 45 $s$','T = 55 $s$');
    set(h11,'FontSize',16,'interpreter','latex','FontSize');
    xlabel('Iterations','interpreter','latex','FontSize',16);
    ylabel('Relative decrease','interpreter','latex','FontSize',16);
    
    saveas(gcf,[dirname1,'/Outer.fig']);
    print ([dirname1,'/Outer.eps'], '-depsc2', '-r600')
end


