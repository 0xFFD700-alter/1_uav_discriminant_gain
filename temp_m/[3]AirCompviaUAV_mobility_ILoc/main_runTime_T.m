clear;clc;
addpath('./afunction');
dname = ['runTimeT_txt'];
mkdir (dname);
delete ([dname,'/*.txt'])
dirname1 = 'runTimeT_test';
mkdir (dirname1);

%% para
Generate_parameter
para.dname = dname; 

T_set = 20:20:100; 
len = length(T_set);
[obj1,obj2,time1,time2] = deal(zeros(len,1));

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
    tic
    [Variable1,allobj1,err1,~,~] = BCD_ADMM_Block3(Ksize,Location,q_strFloc,c_IF,para1);
    t1 = toc;
    time1(ix) = t1; 
    temp = allobj1;
     obj1(ix, 1) = temp(end);
    save([dirname,'/',['method',num2str(ik)]],['Variable',num2str(ik)],['allobj',num2str(ik)],['err',num2str(ik)],'t1');
    save([dirname1,'/',['method',num2str(ik)]],['obj',num2str(ik)],['time',num2str(ik)]);
end
    
ik = 2;
for ix = 1:len
    dirname2 = num2str(ix);
    dirname = [dirname1,'/',dirname2];
    load([dirname,'/',['parameter']])
    tic
    [Variable2,allobj2,err2] = BCD_SCA_Block3(Ksize,Location,q_strFloc,c_IF,para);
    t2 = toc;
    time2(ix) = t2;
    temp = allobj2;
    obj2(ix, 1) = temp(end);
    save([dirname,'/',['method',num2str(ik)]],['Variable',num2str(ik)],['allobj',num2str(ik)],['err',num2str(ik)],'t2');
    save([dirname1,'/',['method',num2str(ik)]],['obj',num2str(ik)],['time',num2str(ik)]);
end


   
   
   
    
   
  
   
   


 

flog1 =0;
if flog1
    
    close;  figure(1)
    vals = [time1, time2];
    b = bar(T_set,vals);
    b(1).FaceColor = [0.8500 0.3250 0.0980];
    b(2).FaceColor = [0 0.4470 0.7410];
    
    b(2).LineStyle = '--';
    b(1).LineStyle = '-';
    
    b(2).LineWidth = 1;
    b(1).LineWidth = 1;
    xlabel('Mission time $T (s)$','interpreter','latex','FontSize',16)
    ylabel('Simulation time (s)','interpreter','latex','FontSize',16)
    legend({'Proposed BCD-ADMM','BCD-SCA'},'interpreter','latex','FontSize',16);
    
    saveas(gcf,[dirname1,'/TimeT.fig']);
    print ([dirname1,'/TimeT.eps'], '-depsc2', '-r600')


end


