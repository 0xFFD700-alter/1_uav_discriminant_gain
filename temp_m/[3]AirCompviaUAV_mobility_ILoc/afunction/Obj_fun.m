function [obj,f4_r] = Obj_fun(Ksize,Location,power_p,Denoise_eta,Traj_q)

[K,N,~,~,H,alpha,sigma,beta0,~,~,~] = deal(Ksize{:});
q_time = Traj_q(2:N+1,:);
x = Location.userX; y = Location.userY; coord_w = [x,y];
Cons_matrix = power_p*beta0./repmat(Denoise_eta,1,K); 
q_dim3 = reshape(q_time',1,2,size(q_time,1)); q_rep = repmat(q_dim3,K,1,1);
f1_r = sum(abs(q_rep-coord_w).^2,2); f1_r = (reshape(f1_r,K,size(q_time,1)))';
d_sqdis = f1_r + H^2;   
f2_r = (sqrt(Cons_matrix)./((d_sqdis).^(alpha/4))-1).^2;
f3_r = sigma./Denoise_eta;    
f4_r = (sum(f2_r,2) + f3_r(:))/K^2;
obj = sum(f4_r)/N;
end

