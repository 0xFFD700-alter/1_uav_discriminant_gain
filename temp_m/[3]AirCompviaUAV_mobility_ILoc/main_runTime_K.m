clear;clc;
addpath('./afunction');
dname = ['runTimeK_txt'];
mkdir (dname);
delete ([dname,'/*.txt'])
dirname1 = 'runTimeK_test';
mkdir (dirname1);

%% para
Generate_parameter
para.dname = dname; 

K_set = 40:10:80; 
len = length(K_set);
numTest = 10;

%% location

[obj1,obj2,time1,time2] = deal(zeros(len,numTest));

save( [dirname1,'/data.mat'])
[Location,q_str,q_strFloc] = deal(cell(numTest,1));

for ix = 1:len
    K = K_set(ix);
    num = floor(K*0.3);
    Loc.num = num;
    p1 = P_set(1,1); p2 = P_set(1,2);
    peak_P = [p1*ones(num,1);p2*ones(K-num,1)];
    
    P2_set = P_set.*[N/2,N/2];
    p3 = P2_set(1,1); p4 = P2_set(1,2);
    Aver_P = [p3*ones(num,1);p4*ones(K-num,1)]/N;
    
    Ksize{1} = K;
    Ksize{3} = peak_P;
    Ksize{4} = Aver_P;
     
    dirname2 = num2str(ix);
    dirname = [dirname1,'/',dirname2];
    mkdir (dirname);
    for iNum = 1:numTest

     Location{iNum} = Location_user(Ksize,Loc,[],'mobility');
     q_strFloc{iNum} = initial_straight_lastLoc(Ksize,c_IF,Location{iNum});
    end
    save([dirname,'/',['parameter']],'Ksize','dirname','ix','K','Location','q_strFloc');      
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
    for iNum = 1:numTest
     tic
    [Variable1,allobj1,err1,~,~] = BCD_ADMM_Block3(Ksize,Location{iNum},q_strFloc{iNum},c_IF,para1);
     t1 = toc;
    time1(ix,iNum) = t1; 
    temp = allobj1;
     obj1(ix, iNum) = temp(end);
    end
    save([dirname,'/',['method',num2str(ik)]],['Variable',num2str(ik)],['allobj',num2str(ik)],['err',num2str(ik)],'t1');
    save([dirname1,'/',['method',num2str(ik)]],['obj',num2str(ik)],['time',num2str(ik)]);
end
    
ik = 2;
for ix = 1:len
    dirname2 = num2str(ix);
    dirname = [dirname1,'/',dirname2];
    load([dirname,'/',['parameter']])
    parfor iNum = 1:numTest   
    tic
    [Variable2,allobj2,err2] = BCD_SCA_Block3(Ksize,Location{iNum},q_strFloc{iNum},c_IF,para);
    t2 = toc;
    time2(ix,iNum) = t2;
    temp = allobj2;
    obj2(ix, iNum) = temp(end);
    end
    save([dirname,'/',['method',num2str(ik)]],['Variable',num2str(ik)],['allobj',num2str(ik)],['err',num2str(ik)],'t2');
    save([dirname1,'/',['method',num2str(ik)]],['obj',num2str(ik)],['time',num2str(ik)]);
end

flog1 =0;
if flog1
    close; figure(1)
    vals = [mean(time1,2), mean(time2,2)];
    b = bar(K_set,vals);
    b(2).FaceColor = [0 0.4470 0.7410];
    b(1).FaceColor = [0.8500 0.3250 0.0980];
    b(2).LineStyle = '--';
    b(1).LineStyle = '-';
    
    b(2).LineWidth = 1;
    b(1).LineWidth = 1;
       
    xlabel('Number of sensors $K$','interpreter','latex','FontSize',16)
    ylabel('Simulation time (s)','interpreter','latex','FontSize',16)
    legend({'Proposed BCD-ADMM','BCD-SCA'},'interpreter','latex','FontSize',16);
    
    saveas(gcf,[dirname1,'/TimeK.fig']);
    print ([dirname1,'/TimeK.eps'], '-depsc2', '-r600')


end


