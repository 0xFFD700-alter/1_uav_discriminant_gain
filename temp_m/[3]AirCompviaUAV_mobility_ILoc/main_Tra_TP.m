

clear;clc;
addpath('./afunction');

dname = ['trajTP_txt'];
mkdir (dname);
delete ([dname,'/*.txt'])

dirname1 = ['test_trajTP'];
mkdir (dirname1);
%% setting
Generate_parameter
para.dname = dname; 
T_set = 50;
P_set = [1e-2,5e-3]; 
p1 = P_set(1,1); p2 = P_set(1,2);
peak_P = [p1*ones(num,1);p2*ones(K-num,1)];
Ksize{3} = peak_P;  
%% location
Loc.angle = [pi/2;pi/2+pi/6]; Loc.v = [5,4];
%%
len = length(T_set);

%% Different Scheme
scheme = 3;
for ik = 1:scheme
    eval(['obj',num2str(ik),'=','zeros(len,1)',';']);        
end
[Location,q_str] = deal(cell(length(T_set),1));
LocationX = Location_user(Ksize,Loc,[],'no');

save( [dirname1,'/data.mat'])
%%
for ix = 1:len
    N = T_set(ix)/delta ;
    Ksize{2} = N;   
    P2_set = P_set.*[N/2,N/2];
    p3 = P2_set(1,1); p4 = P2_set(1,2);
    Aver_P = [p3*ones(num,1);p4*ones(K-num,1)]/N;
    Ksize{4} = Aver_P;
    Location = Location_user(Ksize,Loc,LocationX,'mobility');
    q_strFloc = initial_straight_lastLoc(Ksize,c_IF,Location);
    q_str = initial_straight(Ksize,c_IF,Location);
    dirname2 = num2str(ix);
    dirname = [dirname1,'/',dirname2];
    mkdir (dirname); 
    save([dirname,'/',['parameter']],'Ksize','dirname','ix','N','Location','q_strFloc','q_str');    
end


%% Algorithm
%% 1: Three blocks: BCD_ADMM
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
    [Variable1,allobj1,err1,err_q1,err_obj1] = BCD_ADMM_Block3(Ksize,Location,q_str,c_IF,para1);
    temp = allobj1;
    obj1(ix, 1) = temp(end);
    [A_P1,~,S_P1,~] = Obj_fun_power(Ksize,Loc,Location, Variable1);
    save([dirname,'/',['method',num2str(ik)]],['Variable',num2str(ik)],['allobj',num2str(ik)],['err',num2str(ik)]);
    save([dirname,'/',['method',num2str(ik)]],['err_q',num2str(ik)],['err_obj',num2str(ik)],'-append');
    save([dirname,'/',['method',num2str(ik)]],['A_P',num2str(ik)],['S_P',num2str(ik)],'-append');
    save([dirname1,'/',['method',num2str(ik)]],['obj',num2str(ik)]);
   
end

%%     3: full power transmission
ik = 3;
for ix = 1:len
    dirname2 = num2str(ix);
    dirname = [dirname1,'/',dirname2];
    load([dirname,'/',['parameter']])
    
    [Variable3,allobj3,err3] = BCD_SCA_Block2_fixP(Ksize,Location,q_str,c_IF,para);
    temp = allobj3;
    obj3(ix, 1) = temp(end);
    
    [A_P3,~,S_P3,~] = Obj_fun_power(Ksize,Loc,Location, Variable3);
          
    save([dirname,'/',['method',num2str(ik)]],['Variable',num2str(ik)],['allobj',num2str(ik)],['err',num2str(ik)]);
    save([dirname,'/',['method',num2str(ik)]],['A_P',num2str(ik)],['S_P',num2str(ik)],'-append');
    save([dirname1,'/',['method',num2str(ik)]],['obj',num2str(ik)]);
   
end


%%  %%%%%%%%%%%%%%% 2: three blocks: BCD_SCA 
ik = 2;
para2 = para;
para2.maxiter = 200;
q0 = repmat(c_IF,N+1,1);
for ix = 1:len
    dirname2 = num2str(ix);
    dirname = [dirname1,'/',dirname2];
    load([dirname,'/',['parameter']])
   
    [Variable2,allobj2,err2] = BCD_SCA_Block3(Ksize,Location,q_str,c_IF,para2);
    temp = allobj2;
    obj2(ix, 1) = temp(end);
    [A_P2,~,S_P2,~] = Obj_fun_power(Ksize,Loc,Location, Variable2);
        
    save([dirname,'/',['method',num2str(ik)]],['Variable',num2str(ik)],['allobj',num2str(ik)],['err',num2str(ik)]);
    save([dirname,'/',['method',num2str(ik)]],['A_P',num2str(ik)],['S_P',num2str(ik)],'-append');
    save([dirname1,'/',['method',num2str(ik)]],['obj',num2str(ik)]);    
end




flog1 =0;
if flog1
    nn = 15;
    close; figure(1)
    ix = 1;
    dirname2 = num2str(ix);
    dirname = [dirname1,'/',dirname2];
    load([dirname,'/',['parameter']])
    load([dirname,'/',['method1']])  
   load([dirname,'/',['method2']]) 
   load([dirname,'/',['method3']]) 
    q1 = Variable1.q;
    q3 = Variable3.q;
    q2 = Variable2.q;
    plot([c_IF(1,1);q3(2:nn:end,1);q3(end,1)],[c_IF(1,2);q3(2:nn:end,2);q3(end,2)],'--<','LineWidth',2,'MarkerSize',6); hold on
    plot([c_IF(1,1);q2(2:nn:end,1);q2(end,1)],[c_IF(1,2);q2(2:nn:end,2);q2(end,2)],'-.d','LineWidth',2,'MarkerSize',6); hold on
    plot([c_IF(1,1);q1(2:nn:end,1);q1(end,1)],[c_IF(1,2);q1(2:nn:end,2);q1(end,2)],'-p','LineWidth',2,'MarkerSize',6); hold on
    
    
    h1 = plot(c_IF(1,1),c_IF(1,2),'p','Color','red','MarkerSize',8); hold on
    set(h1,'MarkerFaceColor',get(h1,'color'));
    plot(Location.userX,Location.userY,'s','Color',RGBmatrix(1,:),'MarkerSize',6); hold on
    center_mobility = Location.center_mobility;
    
    for i = 1:part
        plot(center_mobility(1:nn:end,1,i),center_mobility(1:nn:end,2,i),'-^','Color',RGBmatrix(1,:),'LineWidth',2,'MarkerSize',6); hold on
    end  
    WW = Location.w;
    W1 = WW(N:N:end,:);
    plot(W1(:,1),W1(:,2),'s','Color',RGBmatrix(1,:),'MarkerSize',6); hold on
    grid on;
    set(gca,'GridLineStyle','--','GridColor','k', 'GridAlpha',0.2); 
    xlim([c_region(1,1) c_region(2,1)])
    ylim([c_region(1,2) c_region(2,2)])
    h11 = legend( 'TO w/o PC', 'BCD-SCA','BCD-ADMM' );
    set(h11,'FontSize',14,'interpreter','latex');
    xlabel('$x$(m)','interpreter','latex','FontSize',16);
    ylabel('$y$(m)','interpreter','latex','FontSize',16);
   
    saveas(gcf,[dirname1,'/Traj.fig']);
    print Traj.eps -depsc2 -r600


%% power 
   
    
    close; figure(3)
    p = repmat(min(Aver_P',peak_P'),N,1);
    S_P3 = sum(p,2);
    
     ix = 1;  
     SP1 = S_P1;
     SP2 = S_P2;
     SP3 = S_P3;
     N = length (SP1);
     time = [1:N]*delta;
     plot(time,pow2db(SP3*1e3),'-.','LineWidth',2,'MarkerSize',6); hold on    
     plot(time,pow2db(SP2*1e3),'--','LineWidth',2,'MarkerSize',6); hold on
     plot(time,pow2db(SP1*1e3),'-','LineWidth',2,'MarkerSize',6); hold on  
     plot(time, pow2db(P_set(1,1)*1e3*num + P_set(1,2)*1e3*(K-num))*ones(N,1),':','Color','black','LineWidth',2); hold on     
  
    grid on;
    set(gca,'GridLineStyle','--','GridColor','k', 'GridAlpha',0.2);  
    h11 = legend( 'TO w/o PC', 'BCD-SCA','BCD-ADMM','Upper bound');
    set(h11,'interpreter','latex','FontSize',14);
    yticks(20:26)
    ylim([20 26])
    xlabel('Time $t$ (s)','interpreter','latex','FontSize',16);
    ylabel('Transmit power (dBm)','interpreter','latex','FontSize',16);

    saveas(gcf,[dirname1,'/Sp.fig']);
    print Sp.eps -depsc2 -r600
 
    

    
   
end


