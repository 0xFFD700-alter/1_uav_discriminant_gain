function [q_init] = initial_straight_lastLoc(Ksize,c_IF,Location)
[~,N,~,~,~,~,~,~,Vmax,~,delta] = deal(Ksize{:});
center_mobility =  Location.center_mobility;
c_geom = sum(center_mobility(end,:,:),3)/2;
dis = norm(c_geom-c_IF(1,:));
T = N*delta;
q_x = zeros(N+1,1); q_y = zeros(N+1,1);
if dis <= Vmax*T
   T1 =  norm(c_geom-c_IF(1,:))/Vmax;
   N1 = ceil( T1/delta);
  q_x(1:N1,:) = c_IF(1,1) + (c_geom(1,1)-c_IF(1,1))/(N1).*(0:N1-1);
  q_y(1:N1,:) = c_IF(1,2) + (c_geom(1,2)-c_IF(1,2))/(N1).*(0:N1-1);
   
  q_x(N1+1:end,:) = c_geom(1,1);
  q_y(N1+1:end,:) = c_geom(1,2) ;

  q_x(q_x<1e-8) = 0; q_y(q_y<1e-8) = 0;

  q_init = [q_x,q_y];
  
else
    
    A = [c_IF(1,1),1; c_geom(1,1),1 ]; b = [c_IF(1,2);c_geom(1,2)];
    a = linsolve(A,b);
    
    a1 = (a(1))^2+1; b1 = 2*(a(2)-c_IF(1,2) - c_IF(1,1));
    
    c1 = (a(2)-c_IF(1,2))^2+(c_IF(1,1))^2-(Vmax*T)^2;
    
    x1 = roots([a1 b1 c1]);
    x1 = x1(find(min( c_geom(1,1), c_IF(1,1))<= x1 & x1 <= max( c_geom(1,1), c_IF(1,1))));
    y1 = a(1)*x1 + a(2);
    x1(x1<1e-8) = 0; y1(y1<1e-8) = 0;
    cc = [x1,y1];
    
    T1 =  norm(cc-c_IF(1,:))/Vmax;
    N1 = ceil( T1/delta);
    q_x(1:N1+1,:) = c_IF(1,1) + (cc(1,1)-c_IF(1,1))/(N1).*(0:N1);
    q_y(1:N1+1,:) = c_IF(1,2) + (cc(1,2)-c_IF(1,2))/(N1).*(0:N1);
    q_x(q_x<1e-8) = 0; q_y(q_y<1e-8) = 0;
    q_init = [q_x,q_y];
end

