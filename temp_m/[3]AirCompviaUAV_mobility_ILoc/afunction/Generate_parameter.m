
RGBmatrix = [0.04, 0.09, 0.27;  %black 
             0.89,0.09,0.05; %red
              0.8500 0.3250 0.0980; %orange 
              1,0.6,0.07; % yellon 
              0.24,0.57,0.25;  %green
             0 0.4470 0.7410;]; %blue   

para.outerverb = 1;     
para.innerverb = 0;
para.maxiter = 100;
para.flag = 0; %% ADMM_CVX
para.outertol = 1e-3;
para.SCAscal = 10;



K = 50; T = 50; sigma = db2pow(-80)/10^3; numTest = 100; 
Vmax = 20; Vmin= 0; beta0 = 1e-4; alpha = 2;  H= 100;  delta = 0.2;
N = T/delta;

part = 2; num = floor(K*0.3);  
c_w = [50,100;350,150]; c_IF = [200,0]; 

c_r = 50; c_region = [0,0;400,400];
Loc.num = num; Loc.part = part; 
Loc.c_w = c_w; Loc.c_r = c_r; 
Loc.c_region = c_region;


P_set = [1e-2,5e-3]; 
p1 = P_set(1,1); p2 = P_set(1,2);
peak_P = [p1*ones(num,1);p2*ones(K-num,1)];

Ksize = num2cell([K,1,1,1,H,alpha,sigma,beta0,Vmax,Vmin,delta]);
Ksize{3} = peak_P;  
Ksize{2} = N;