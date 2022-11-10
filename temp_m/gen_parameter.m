load('./data/inference/mean.mat', 'mu'); % mean value
load('./data/inference/var.mat', 'sigma'); % ground-true var
load('./data/inference/var_sensor.mat', 'delta'); % distortion var
load('./data/inference/trajectory_sensor.mat', 'w'); % trajectory of sensors
load('./data/inference/trajectory_sensor_centroid.mat', 'centroid'); % trajectory centroids of sensors
load('./data/inference/data_test_noise.mat', 'z'); % distorted data
load('./data/inference/label_test.mat', 'label_test'); % label of test data


% data and model to perform inference tasks
infer.L = size(mu, 1);
infer.num_test_samples = size(z, 1);
infer.z = z;
infer.label_test = label_test;


% data dimensions
dim.N = size(mu, 2);                % N -> # of feature dims (# of time slots)
dim.K = size(delta, 1);             % K -> # of sensors
num_a = fix(dim.K * 0.3) + 1;
num_b = dim.K - num_a;


% parameters of data distribution (used to compute discriminant gain)
gain.u = zeros(1, dim.N);
for i = 1: infer.L - 1
    for j = i + 1: infer.L
        gain.u = gain.u + (mu(i, :) - mu(j, :)) .^ 2;
    end
end
gain.u = gain.u * 2 / infer.L / (infer.L - 1);          % average square mean
gain.sigma = sigma;                         % variance of ground truth
gain.delta = delta;                         % variance of distortion
gain.delta_0 = 1e-10;                       % variance of Gaussian noise


% power constraints
power.ratio = 0.5;                                      % ratio of the average to the peak
power.L_0 = 1e-4;                                       % channel fading at reference distance (1m)
power.E = gain.delta + gain.sigma + mean(mu.^2, 1);     % expectation of signal power
power.H = 50.0;                                        % UAV hovering altitude
power.w = w;                                            % trajectory of sensors


% UAV mobility constraints
uav.slot = 50.0 / dim.N;            % duration of each time slot (duration / # of time slots)
uav.Vm = 30.0;                      % UAV maximum speed
uav.q_init = [200.0 0.0];           % UAV initial position


% alternating opt initialization, q_iter -> init UAV trajectory
q_iter = zeros(dim.N, 2);
if norm(centroid(1, :) - uav.q_init) <= uav.slot * uav.Vm
    q_iter(1, :) = centroid(1, :);
else
    q_iter(1, :) = uav.q_init + (centroid(1, :) - uav.q_init) * uav.slot * uav.Vm / norm(centroid(1, :) - uav.q_init);
end
for i = 2:dim.N
    if norm(centroid(i, :) - q_iter(i - 1, :)) <= uav.slot * uav.Vm
        q_iter(i, :) = centroid(i, :);
    else
        q_iter(i, :) = q_iter(i - 1, :) + (centroid(i, :) - q_iter(i - 1, :)) * uav.slot * uav.Vm / norm(centroid(1, :) - q_iter(i - 1, :));
    end
end


% sca opt parameter settings
sca.momentum = 0.8;
sca.epsilon = 1e-2;
sca.patience = 5;


% opt parameter settings
epsilon = 1e-2;


% auxiliary variables for alternating opt
repeat = 10;
% gain_fun = @(x) sum(sum(x) .^ 2 .* gain.u ./ (gain.sigma .* sum(x) .^ 2 + sum(x .^ 2 .* gain.delta) + gain.delta_0));

% scalars
init_scale = 1e3;
scale_c = 1e10;
scale_q = 1e0;
