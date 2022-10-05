function [plAU,Location] = Channel_LargeFading(Ksize,Location,q_time)
%K: number of user

[K,N,~,~,H,alpha,~,beta0,~,~,~] = deal(Ksize{:});
%% loss based on distance
w = Location.w;

X = repmat(eye(N),K,1); X = sparse(X);
q_rep = X*q_time;

d_2D = sum(abs(q_rep-w).^2,2);
d_dis = sqrt(d_2D + H^2); 
plAU = beta0 * (d_dis).^(-alpha); 
plAU  = reshape(plAU,N,K);

end

