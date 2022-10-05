clear;clc;
addpath('./afunction');
dname = ['noise_txt2'];
mkdir (dname);
delete ([dname,'/*.txt'])
dirname1 = ['noise_test2'];
mkdir (dirname1);

%% generate parameter
Generate_parameter
para.dname = dname; 
numTest = 50;
sigma_set = -95:5:-60; % dbm
len = length(sigma_set);
%% location
[Location,LocationX,q_str,q_strFloc] = deal(cell(numTest,1));
for iNum = 1:numTest
    LocationX{iNum} = Location_user(Ksize,Loc,[],'no');
end
%% Different Scheme
scheme = 6;
for ik = 1:scheme
    eval(['obj',num2str(ik),'=','zeros(len,numTest)',';']); 
    eval(['SP',num2str(ik),'=','zeros(len,numTest)',';']); 
    eval(['Meanobj',num2str(ik),'=','zeros(len,1)',';']); 
    eval(['MeanSP',num2str(ik),'=','zeros(len,1)',';']); 
    eval(['Variable',num2str(ik),'=','cell(numTest,1)',';']); 
    eval(['allobj',num2str(ik),'=','cell(numTest,1)',';']); 
    eval(['err',num2str(ik),'=','cell(numTest,1)',';']);  
    eval(['A_P',num2str(ik),'=','cell(numTest,1)',';']);
    eval(['S_P',num2str(ik),'=','cell(numTest,1)',';']); 
end

save( [dirname1,'/data.mat'])
%%

T = 50; N = T/delta; Ksize{2} = N;
p1 = P_set(1,1); p2 = P_set(1,2);
peak_P = [p1*ones(num,1);p2*ones(K-num,1)];
P2_set = P_set.*[N/2,N/2];
p3 = P2_set(1,1); p4 = P2_set(1,2);
Aver_P = [p3*ones(num,1);p4*ones(K-num,1)]/N;
Ksize{3} = peak_P;
Ksize{4} = Aver_P;

for iNum = 1:numTest
    Location{iNum} = Location_user(Ksize,Loc,LocationX{iNum},'mobility');
    q_str{iNum} = initial_straight(Ksize,c_IF,Location{iNum});
    q_strFloc{iNum,1} = initial_straight_lastLoc(Ksize,c_IF,Location{iNum});
end
for ix = 1:len
    sigma = db2pow(sigma_set(ix))/10^3;
    Ksize{7} = sigma;   
    dirname2 = num2str(ix);
    dirname = [dirname1,'/',dirname2];
    mkdir (dirname);
    
    save([dirname,'/',['parameter']],'Ksize','dirname','ix','q_str','sigma','Location','q_strFloc');
end


%% 4 Fly-hover
ik = 4;
for ix = 1:len
    dirname2 = num2str(ix);
    dirname = [dirname1,'/',dirname2];
    load([dirname,'/',['parameter']])
    parfor iNum = 1:numTest
        [Variable4{iNum},allobj4{iNum},err4{iNum}] = BCD_Block2_fixQ(Ksize,Location{iNum},q_str{iNum},c_IF,para);
        temp = allobj4{iNum};
        obj4(ix, iNum) = temp(end);
        [A_P4{iNum},~,S_P4{iNum},~] = Obj_fun_power(Ksize,Loc,Location{iNum}, Variable4{iNum});
        SP4(ix, iNum) = sum(S_P4{iNum});
    end
    tempobj = eval(['obj',num2str(ik)]);
    eval(['Meanobj',num2str(ik),'(ix,1)','=','mean(tempobj(ix,:))',';']);
    eval(['meanObj',num2str(ik),'=','mean(tempobj(ix,:))',';']);
    tempobj1 = eval(['SP',num2str(ik)]);
    eval(['MeanSP',num2str(ik),'(ix,1)','=','mean(tempobj1(ix,:))',';']);
    eval(['meanSP',num2str(ik),'=','mean(tempobj1(ix,:))',';']);
    
    save([dirname,'/',['method',num2str(ik)]],['Variable',num2str(ik)],['allobj',num2str(ik)],['err',num2str(ik)]);
    save([dirname,'/',['method',num2str(ik)]],['S_P',num2str(ik)],['A_P',num2str(ik)],['meanObj',num2str(ik)],['meanSP',num2str(ik)],'-append');   
end
save([dirname1,'/',['method',num2str(ik)]],['Meanobj',num2str(ik)],['MeanSP',num2str(ik)]);
save([dirname1,'/',['method',num2str(ik)]],['obj',num2str(ik)],['SP',num2str(ik)],'-append');


%% 5: static
ik = 5;
for ix = 1:len
    dirname2 = num2str(ix);
    dirname = [dirname1,'/',dirname2];
    load([dirname,'/',['parameter']])
    parfor iNum = 1:numTest
        [Variable5{iNum},allobj5{iNum},err5{iNum}] =  BCD_Block2_static(Ksize,Location{iNum},c_IF,c_IF,para);
        temp = allobj5{iNum};
        obj5(ix, iNum) = temp(end);
        [A_P5{iNum},~,S_P5{iNum},~] = Obj_fun_power(Ksize,Loc,Location{iNum}, Variable5{iNum});
        SP5(ix, iNum) = sum(S_P5{iNum});
    end
    tempobj = eval(['obj',num2str(ik)]);
    eval(['Meanobj',num2str(ik),'(ix,1)','=','mean(tempobj(ix,:))',';']);
    eval(['meanObj',num2str(ik),'=','mean(tempobj(ix,:))',';']);
    tempobj1 = eval(['SP',num2str(ik)]);
    eval(['MeanSP',num2str(ik),'(ix,1)','=','mean(tempobj1(ix,:))',';']);
    eval(['meanSP',num2str(ik),'=','mean(tempobj1(ix,:))',';']);
    
    save([dirname,'/',['method',num2str(ik)]],['Variable',num2str(ik)],['allobj',num2str(ik)],['err',num2str(ik)]);
    save([dirname,'/',['method',num2str(ik)]],['S_P',num2str(ik)],['A_P',num2str(ik)],['meanObj',num2str(ik)],['meanSP',num2str(ik)],'-append');    


end
save([dirname1,'/',['method',num2str(ik)]],['Meanobj',num2str(ik)],['MeanSP',num2str(ik)]);
save([dirname1,'/',['method',num2str(ik)]],['obj',num2str(ik)],['SP',num2str(ik)],'-append');


%% %%%%%%%%%%%%%%% 6: fly-hover last Floc
ik = 6;
for ix = 1:len
    dirname2 = num2str(ix);
    dirname = [dirname1,'/',dirname2];
    load([dirname,'/',['parameter']])
    parfor iNum = 1:numTest
        [Variable6{iNum},allobj6{iNum},err6{iNum}] = BCD_Block2_fixQ(Ksize,Location{iNum},q_strFloc{iNum},c_IF,para);
        temp = allobj6{iNum};
        obj6(ix, iNum) = temp(end);
        [A_P6{iNum},~,S_P6{iNum},~] = Obj_fun_power(Ksize,Loc,Location{iNum}, Variable6{iNum});
        SP6(ix, iNum) = sum(S_P6{iNum});
    end
    tempobj = eval(['obj',num2str(ik)]);
    eval(['Meanobj',num2str(ik),'(ix,1)','=','mean(tempobj(ix,:))',';']);
    eval(['meanObj',num2str(ik),'=','mean(tempobj(ix,:))',';']);
    tempobj1 = eval(['SP',num2str(ik)]);
    eval(['MeanSP',num2str(ik),'(ix,1)','=','mean(tempobj1(ix,:))',';']);
    eval(['meanSP',num2str(ik),'=','mean(tempobj1(ix,:))',';']);
    
    save([dirname,'/',['method',num2str(ik)]],['Variable',num2str(ik)],['allobj',num2str(ik)],['err',num2str(ik)]);
    save([dirname,'/',['method',num2str(ik)]],['S_P',num2str(ik)],['A_P',num2str(ik)],['meanObj',num2str(ik)],['meanSP',num2str(ik)],'-append');   
end
save([dirname1,'/',['method',num2str(ik)]],['Meanobj',num2str(ik)],['MeanSP',num2str(ik)]);
save([dirname1,'/',['method',num2str(ik)]],['obj',num2str(ik)],['SP',num2str(ik)],'-append');


%% %%%%%%%%%%%%%%% 1: Three blocks: BCD_ADMM
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
parfor iNum = 1:numTest
    [Variable1{iNum},allobj1{iNum},err1{iNum},~,~] = BCD_ADMM_Block3(Ksize,Location{iNum},q_strFloc{iNum},c_IF,para1);
    temp = allobj1{iNum};
    obj1(ix, iNum) = temp(end);
    [A_P1{iNum},~,S_P1{iNum},~] = Obj_fun_power(Ksize,Loc,Location{iNum}, Variable1{iNum});
    SP1(ix, iNum) = sum(S_P1{iNum});
end
    tempobj = eval(['obj',num2str(ik)]);
    eval(['Meanobj',num2str(ik),'(ix,1)','=','mean(tempobj(ix,:))',';']);
    eval(['meanObj',num2str(ik),'=','mean(tempobj(ix,:))',';']);
    tempobj1 = eval(['SP',num2str(ik)]);
    eval(['MeanSP',num2str(ik),'(ix,1)','=','mean(tempobj1(ix,:))',';']);
    eval(['meanSP',num2str(ik),'=','mean(tempobj1(ix,:))',';']);
    
    save([dirname,'/',['method',num2str(ik)]],['Variable',num2str(ik)],['allobj',num2str(ik)],['err',num2str(ik)]);
    save([dirname,'/',['method',num2str(ik)]],['S_P',num2str(ik)],['A_P',num2str(ik)],['meanObj',num2str(ik)],['meanSP',num2str(ik)],'para1','-append');    

    
    save([dirname1,'/',['method',num2str(ik)]],['Meanobj',num2str(ik)],['MeanSP',num2str(ik)]);
    save([dirname1,'/',['method',num2str(ik)]],['obj',num2str(ik)],['SP',num2str(ik)],'-append');
end
save([dirname1,'/',['method',num2str(ik)]],['Meanobj',num2str(ik)],['MeanSP',num2str(ik)]);
save([dirname1,'/',['method',num2str(ik)]],['obj',num2str(ik)],['SP',num2str(ik)],'-append');


 %% %%%%%%%%%%%%%%%%%%% 3: full power 
ik = 3;
for ix = 1:len
    dirname2 = num2str(ix);
    dirname = [dirname1,'/',dirname2];
    load([dirname,'/',['parameter']])
    parfor iNum = 1:numTest
        [Variable3{iNum},allobj3{iNum},err3{iNum}] = BCD_SCA_Block2_fixP(Ksize,Location{iNum},q_strFloc{iNum},c_IF,para);
        temp = allobj3{iNum};
        obj3(ix, iNum) = temp(end);
        [A_P3{iNum},~,S_P3{iNum},~] = Obj_fun_power(Ksize,Loc,Location{iNum}, Variable3{iNum});
        SP3(ix, iNum) = sum(S_P3{iNum});
    end
    tempobj = eval(['obj',num2str(ik)]);
    eval(['Meanobj',num2str(ik),'(ix,1)','=','mean(tempobj(ix,:))',';']);
    eval(['meanObj',num2str(ik),'=','mean(tempobj(ix,:))',';']);
    tempobj1 = eval(['SP',num2str(ik)]);
    eval(['MeanSP',num2str(ik),'(ix,1)','=','mean(tempobj1(ix,:))',';']);
    eval(['meanSP',num2str(ik),'=','mean(tempobj1(ix,:))',';']);
    
    save([dirname,'/',['method',num2str(ik)]],['Variable',num2str(ik)],['allobj',num2str(ik)],['err',num2str(ik)]);
    save([dirname,'/',['method',num2str(ik)]],['S_P',num2str(ik)],['A_P',num2str(ik)],['meanObj',num2str(ik)],['meanSP',num2str(ik)],'-append');
 
    
    save([dirname1,'/',['method',num2str(ik)]],['Meanobj',num2str(ik)],['MeanSP',num2str(ik)]);
    save([dirname1,'/',['method',num2str(ik)]],['obj',num2str(ik)],['SP',num2str(ik)],'-append');
end
save([dirname1,'/',['method',num2str(ik)]],['Meanobj',num2str(ik)],['MeanSP',num2str(ik)]);
save([dirname1,'/',['method',num2str(ik)]],['obj',num2str(ik)],['SP',num2str(ik)],'-append');     


%%  %%%%%%%%%%%%%%% 2: three blocks: BCD_SCA 
ik = 2;
for ix = 1:len
    dirname2 = num2str(ix);
    dirname = [dirname1,'/',dirname2];
    load([dirname,'/',['parameter']])
    parfor iNum = 1:numTest
        [Variable2{iNum},allobj2{iNum},err2{iNum}] = BCD_SCA_Block3(Ksize,Location{iNum},q_strFloc{iNum},c_IF,para);
        temp = allobj2{iNum};
        obj2(ix, iNum) = temp(end);
        [A_P2{iNum},~,S_P2{iNum},~] = Obj_fun_power(Ksize,Loc,Location{iNum}, Variable2{iNum});
        SP2(ix, iNum) = sum(S_P2{iNum});
    end
    tempobj = eval(['obj',num2str(ik)]);
    eval(['Meanobj',num2str(ik),'(ix,1)','=','mean(tempobj(ix,:))',';']);
    eval(['meanObj',num2str(ik),'=','mean(tempobj(ix,:))',';']);
    tempobj1 = eval(['SP',num2str(ik)]);
    eval(['MeanSP',num2str(ik),'(ix,1)','=','mean(tempobj1(ix,:))',';']);
    eval(['meanSP',num2str(ik),'=','mean(tempobj1(ix,:))',';']);
    
    save([dirname,'/',['method',num2str(ik)]],['Variable',num2str(ik)],['allobj',num2str(ik)],['err',num2str(ik)]);
    save([dirname,'/',['method',num2str(ik)]],['S_P',num2str(ik)],['A_P',num2str(ik)],['meanObj',num2str(ik)],['meanSP',num2str(ik)],'-append');
   
    
    save([dirname1,'/',['method',num2str(ik)]],['Meanobj',num2str(ik)],['MeanSP',num2str(ik)]);
    save([dirname1,'/',['method',num2str(ik)]],['obj',num2str(ik)],['SP',num2str(ik)],'-append');
end
save([dirname1,'/',['method',num2str(ik)]],['Meanobj',num2str(ik)],['MeanSP',num2str(ik)]);
save([dirname1,'/',['method',num2str(ik)]],['obj',num2str(ik)],['SP',num2str(ik)],'-append');
 
 

%% %% plot
flog1 =0;
if flog1
    close; 
    figure(1)
    ii = 1:6;
    semilogy(sigma_set(ii),Meanobj5(ii,:),'--^','LineWidth',2,'MarkerSize',6); hold on  
    semilogy(sigma_set(ii),Meanobj3(ii,:),'--V','LineWidth',2,'MarkerSize',6); hold on
    semilogy(sigma_set(ii),Meanobj6(ii,:),'-.^','LineWidth',2,'MarkerSize',6); hold on
    semilogy(sigma_set(ii),Meanobj2(ii,:),'-->','LineWidth',2,'MarkerSize',6); hold on
    semilogy(sigma_set(ii),Meanobj1(ii,:),'-^','LineWidth',2,'MarkerSize',6); hold on  
    grid on;
    set(gca,'GridLineStyle','--','GridColor','k', 'GridAlpha',0.2);  
     h11 = legend('Static UAV','TO w/o PC','Fly-hover  w/ PC',...
    'BCD-SCA','BCD-ADMM','interpreter','latex');
    set(h11,'FontSize',14);
  %  ylim([1.3*10^-3 2.5*10^-3])
    xlabel('$\sigma^2$ (dBm)','interpreter','latex','FontSize',14);
    ylabel('Time-averaing MSE','interpreter','latex','FontSize',14);
    
 
    saveas(gcf,[dirname1,'/AlogNoise.fig']);
    print ([dirname1,'/AlogNoise.eps'], '-depsc2', '-r600')
end


