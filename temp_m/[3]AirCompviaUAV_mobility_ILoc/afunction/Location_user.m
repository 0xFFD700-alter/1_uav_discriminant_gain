function [Location] = Location_user(Ksize,Loc,Location,flag)
%K: number of user
[K,N,~,~,~,~,~,~,~,~,delta] = deal(Ksize{:});

part = Loc.part; num = Loc.num;
c_w = Loc.c_w; c_r = Loc.c_r; c_region = Loc.c_region ;

if ~isfield(Location,'userX')   
    [x,y] = deal(zeros(K,1));
    c_geom_part = zeros(part,2); v = zeros(part,1); angle1 = zeros(part,1);
    for iM = 1: part
        if iM == part && part > 1
            K_index = (iM-1)*num+1: K;
            angle1(iM) = rand*pi/2 + pi/2;
        else
            K_index = (iM-1)*num+1: (iM)*num;
            angle1(iM) = rand*pi/2;
        end
        %% Square region
        r_p = rand(length(K_index),1)*c_r;
        angle = rand(length(K_index),1)*2*pi;
        x(K_index,1) = c_w(iM,1) + r_p.*cos(angle);
        y(K_index,1) = c_w(iM,2)  + r_p.*sin(angle);
        c_geom_part(iM,:) = sum([x(K_index,1),y(K_index,1)],1)/length(K_index);      
        v(iM) = rand*7 + 1;
        
    end
    c_geom = sum([x,y],1)/K;
    Location.c_geom_part = c_geom_part;
    Location.c_geom = c_geom;
    Location.userX = x; Location.userY = y;
    Location.v = v; 
    if ~isfield(Loc,'angle')   
        Location.angle = angle1;
        Location.v = v;
    else
        Location.angle = Loc.angle; 
        Location.v = Loc.v;  
    end

end


 if strcmp(flag,'static')  
    x = Location.userX; y =  Location.userY; coord_w = [x,y];
    coord_w1 = (reshape(coord_w',2*K,1))';
    w_B2 = repmat(coord_w1,N,1);
    w_B2 = (mat2cell(w_B2,N,2*ones(1,K)))';
    w_B3 = cell2mat(w_B2);
    Location.w = w_B3;
 end
 if  strcmp(flag,'mobility')
     center_mobility = zeros(N,2,part);
     v = Location.v; angle = Location.angle;
     c_diff = zeros(N*K,2); w = zeros(N*K,2);
     for iM = 1: part
         temp1 = zeros(N,2);
         c = c_w(iM,:);
         temp1(1,:) = c;
        
        for iN = 2: N          
            x = temp1(iN-1,1) + v(iM)*delta*cos(angle(iM));
            y = temp1(iN-1,2)  + v(iM)*delta.*sin(angle(iM));
            if   x-c_r < c_region(1,1) 
                 angle(iM) = rand*pi-pi/2 ;
            elseif x+c_r > c_region(2,1) 
                 angle(iM) =  rand*pi+pi/2;
            elseif y-c_r < c_region(1,2) 
                 angle(iM) =  rand*pi;
            elseif y+c_r > c_region(2,2)
                 angle(iM) =  -rand*pi;             
            end
            x = temp1(iN-1,1) + v(iM)*delta*cos(angle(iM));
            y = temp1(iN-1,2)  + v(iM)*delta.*sin(angle(iM));
            temp1(iN,:) = [x,y];
        end
        center_mobility(:,:,iM) = temp1;
        
        if iM == part && part > 1
            K_index = (iM-1)*num*N+1: K*N;
            cc = repmat(temp1, K-num,1);
            r_p = rand(length(K_index),1)*c_r;
            angle = rand(length(K_index),1)*2*pi;
            w(K_index,1) = cc(:,1) + r_p.*cos(angle);
            w(K_index,2) = cc(:,2)  + r_p.*sin(angle);          
        else
            K_index = (iM-1)*num*N+1: (iM)*num*N;
            cc = repmat(temp1, num,1);
            temp_r_p = rand(length(K_index),1)*c_r;
            temp_angle = rand(length(K_index),1)*2*pi;
            w(K_index,1) = cc(:,1) + temp_r_p.*cos(temp_angle);
            w(K_index,2) = cc(:,2)  + temp_r_p.*sin(temp_angle);  
        end
       
    end 
   
    Location.w = w;
    Location.center_mobility = center_mobility;
end




end

