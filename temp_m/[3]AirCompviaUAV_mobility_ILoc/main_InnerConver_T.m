clear;clc;
addpath('./afunction');
dname = ['InnerT_txt'];
mkdir (dname);
delete ([dname,'/*.txt'])
dirname1 = 'InnerT_test';
mkdir (dirname1);

%% para
Generate_parameter
para.dname = dname; 
para.flag = 1; %% ADMM_CVX
para.innerverb = 1;
para.maxiter = 2;
T_set = 20:20:60; 
len = length(T_set);
obj1 = zeros(len,1);

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
    q_str = initial_straight(Ksize,c_IF,Location);
    save([dirname,'/',['parameter']],'Ksize','dirname','ix','N','Location','q_strFloc','q_str');      
end  

ik = 1;
para1 = para;
para1.scal0 = 1; para1.scal1 = 1; para1.scal2 = 1; para1.scal3 = 1;
rho1 = 1; rho2 = 1; rho3 = 20;
rho = [rho1,rho2,rho3];
para1.rho = rho;

for ix = 1:len
    dirname2 = num2str(ix);
    dirname = [dirname1,'/',dirname2];
    load([dirname,'/',['parameter']])
    [Variable1,allobj1,err1,err_q1,err_obj1] = BCD_ADMM_Block3(Ksize,Location,q_strFloc,c_IF,para1);
    temp = allobj1;
    obj1(ix, 1) = temp(end);
    save([dirname,'/',['method',num2str(ik)]],['Variable',num2str(ik)],['allobj',num2str(ik)],['err',num2str(ik)]);
    save([dirname,'/',['method',num2str(ik)]],['err_q',num2str(ik)],['err_obj',num2str(ik)],'-append');
    save([dirname1,'/',['method',num2str(ik)]],['obj',num2str(ik)]);
end




flog1 =0;
if flog1
    close; figure(1)
    ix = 1;
    dirname2 = num2str(ix);
    dirname = [dirname1,'/',dirname2];
    load([dirname,'/',['parameter']])
    load([dirname,'/',['method1']])
    err_obj = err_obj1{1};
    h1 = semilogy(err_obj(1:272),'-','LineWidth',2);hold on
    ix = 2;
    dirname2 = num2str(ix);
    dirname = [dirname1,'/',dirname2];
    load([dirname,'/',['parameter']])
    load([dirname,'/',['method1']])
    err_obj = err_obj1{1};
    h1 = semilogy(err_obj(1:153),'--','LineWidth',2);hold on  
    ix = 3;
    dirname2 = num2str(ix);
    dirname = [dirname1,'/',dirname2];
    load([dirname,'/',['parameter']])
    load([dirname,'/',['method1']])
    err_obj = err_obj1{1};
    h1 = semilogy(err_obj(1:140),'-.','LineWidth',2);hold on 
    
    
    grid on;
    set(gca,'GridLineStyle','--','GridColor','k', 'GridAlpha',0.2);
    h11 = legend('T = 20 $s$','T = 40 $s$','T = 60 $s$');
    set(h11,'FontSize',16,'interpreter','latex');
    xlabel('Iterations ($j$)','interpreter','latex','FontSize',16);
    ylabel('Relative error','interpreter','latex','FontSize',16);
    
    
    saveas(gcf,[dirname1,'/inner.fig']);
    print inner.eps -depsc2 
    
    
    close; figure(2)
    ix = 1;
    dirname2 = num2str(ix);
    dirname = [dirname1,'/',dirname2];
    load([dirname,'/',['parameter']])
    load([dirname,'/',['method1']])
    err_q = err_q1{2};
    h1 = semilogy(err_q,'-','LineWidth',2);hold on
    ix = 2;
    dirname2 = num2str(ix);
    dirname = [dirname1,'/',dirname2];
    load([dirname,'/',['parameter']])
    load([dirname,'/',['method1']])
    err_q = err_q1{2};
    h1 = semilogy(err_q,'--','LineWidth',2);hold on  
    ix = 3;
    dirname2 = num2str(ix);
    dirname = [dirname1,'/',dirname2];
    load([dirname,'/',['parameter']])
    load([dirname,'/',['method1']])
    err_q = err_q1{2};
    h1 = semilogy(err_q,'-.','LineWidth',2);hold on 
    
    
    grid on;
    set(gca,'GridLineStyle','--','GridColor','k', 'GridAlpha',0.2);
    h11 = legend('T = 20 $s$','T = 40 $s$','T = 60 $s$');
    set(h11,'FontSize',16,'interpreter','latex');
    xlabel('Iterations ($j$)','interpreter','latex','FontSize',16);
    ylabel('Relative error','interpreter','latex','FontSize',16);
    
    
    saveas(gcf,[dirname1,'/inner.fig']);
    print inner.eps -depsc2 
    
end


