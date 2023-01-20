function gen_sensor_trajectory(K, N)
    rng(2020);
    num_a = floor(K * 0.5);
    num_b = K - num_a;
    radius = 25;
    start_a = [50.0 50.0];
    start_b = [450.0 50.0];

    % straight center position
%     end_a = [50.0, 350.0];
%     end_b = [250.0 325.0];
%     centroid_a(:, 1) = linspace(start_a(1), end_a(1), N);
%     centroid_a(:, 2) = linspace(start_a(2), end_a(2), N);
%     centroid_b(:, 1) = linspace(start_b(1), end_b(1), N);
%     centroid_b(:, 2) = linspace(start_b(2), end_b(2), N);

    % random center position
    centroid_a = [start_a; zeros(N - 1, 2)];
    centroid_b = [start_b; zeros(N - 1, 2)];
    region = [0 0; 500 500];
    
    for i = 2:N
        v = 5 + 5 * rand;
        angle = rand * pi;
        x = centroid_a(i - 1, 1) + v * 50 / N * cos(angle);
        y = centroid_a(i - 1, 2) + v * 50 / N * sin(angle);
        if   x - radius < region(1,1)       % hit left
             angle = rand * pi - pi/2 ;
        elseif x + radius > region(2,1)     % hit right
             angle =  rand * pi + pi/2;
        elseif y - radius < region(1,2)     % hit bottom 
             angle =  rand * pi;
        elseif y + radius > region(2,2)     % hit top
             angle =  -rand * pi;             
        end
        centroid_a(i, 1) = centroid_a(i - 1, 1) + v * 50 / N * cos(angle);
        centroid_a(i, 2) = centroid_a(i - 1, 2) + v * 50 / N * sin(angle);
    end
    
    for i = 2:N
        v = 5 + 5 * rand;
        angle = rand * pi;
        x = centroid_b(i - 1, 1) + v * 50 / N * cos(angle);
        y = centroid_b(i - 1, 2) + v * 50 / N * sin(angle);
        if   x - radius < region(1,1)       % hit left
             angle = rand * pi - pi/2 ;
        elseif x + radius > region(2,1)     % hit right
             angle =  rand * pi + pi/2;
        elseif y - radius < region(1,2)     % hit bottom
             angle =  rand * pi;
        elseif y + radius > region(2,2)     % hit top
             angle =  -rand * pi;             
        end
        centroid_b(i, 1) = centroid_b(i - 1, 1) + v * 50 / N * cos(angle);
        centroid_b(i, 2) = centroid_b(i - 1, 2) + v * 50 / N * sin(angle);
    end

    % random steps
    w_a = zeros([num_a N 2]);
    w_b = zeros([num_b N 2]);
    radius_a = rand(num_a, N) * radius;
    angle_a = rand(num_a, N) * 2*pi;
    radius_b = rand(num_b, N) * radius;
    angle_b = rand(num_b, N) * 2*pi;
    w_a(:, :, 1) = centroid_a(:, 1)' + radius_a .* cos(angle_a);
    w_a(:, :, 2) = centroid_a(:, 2)' + radius_a .* sin(angle_a);
    w_b(:, :, 1) = centroid_b(:, 1)' + radius_b .* cos(angle_b);
    w_b(:, :, 2) = centroid_b(:, 2)' + radius_b .* sin(angle_b);

    % % generate linear trajectory
    % ratio = 0.3;
    % num_a = floor(K * 0.3) + 1;
    % num_b = K - num_a;
    % radius = 50;
    % start_a = [50.0 100.0]; end_a = [50.0, 350.0];
    % start_b = [350.0 150.0]; end_b = [250.0 325.0];
    % 
    % % random init position and reshape
    % rad_a = 2 * pi * rand([num_a 1]);
    % dist_a = radius * rand([num_a 1]);
    % rad_b = 2 * pi * rand([num_b 1]);
    % dist_b = radius * rand([num_b 1]);
    % init_a = start_a + [dist_a .* cos(rad_a) dist_a .* sin(rad_a)];
    % init_b = start_b + [dist_b .* cos(rad_b) dist_b .* sin(rad_b)];
    % fin_a = init_a + reshape(end_a - start_a, [1 2]);
    % fin_b = init_b + reshape(end_b - start_b, [1 2]);
    % 
    % % interpolate
    % w_a = zeros([num_a N 2]);
    % w_b = zeros([num_b N 2]);
    % for i = 1:num_a
    %     w_a(i, :, 1) = linspace(init_a(i, 1), fin_a(i, 1), N);
    %     w_a(i, :, 2) = linspace(init_a(i, 2), fin_a(i, 2), N);
    % end
    % for i = 1:num_b
    %     w_b(i, :, 1) = linspace(init_b(i, 1), fin_b(i, 1), N);
    %     w_b(i, :, 2) = linspace(init_b(i, 2), fin_b(i, 2), N);
    % end

    w = [w_a; w_b];
    save('./data/inference/trajectory_sensor.mat', 'w');

    centroid = (centroid_a .* num_a + centroid_b .* num_b) ./ K;
    save('./data/inference/trajectory_sensor_centroid.mat', 'centroid');
    save('./data/inference/trajectory_sensor_centroid_a.mat', 'centroid_a');
    save('./data/inference/trajectory_sensor_centroid_b.mat', 'centroid_b');
end


