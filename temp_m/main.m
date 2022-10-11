% clc;
clear;close all;
load('./data/inference/mean.mat'); % mean value
load('./data/inference/var.mat'); % ground-true var
load('./data/inference/var_sensor.mat'); % distortion var
load('./data/inference/trajectory_sensor.mat'); % trajectory of sensors
load('./data/inference/data_test_noise.mat'); % distorted data
load('./data/inference/label_test.mat'); % label of test data
load('./Mdl.mat'); % SVM model 12-dim



% parameter settings
rng(2022);

% data dimensions
L = size(mu, 1);
dim.N = size(mu, 2);                % N -> # of feature dims (# of time slots)
dim.K = size(delta, 1);             % K -> # of sensors
num_a = fix(dim.K * 0.3) + 1;
num_b = dim.K - num_a;

% parameters of data distribution (used to compute discriminant gain)
gain.u = zeros(1, dim.N);
for i = 1: L - 1
    for j = i + 1: L
        gain.u = gain.u + (mu(i, :) - mu(j, :)) .^ 2;
    end
end
gain.u = gain.u * 2 / L / (L - 1);          % average square mean
gain.sigma = sigma;                         % variance of ground truth
gain.delta = delta;                         % variance of distortion
gain.delta_0 = 1e-11;                       % variance of Gaussian noise

% power constraints
P_list = [150 * ones(num_a, 1); 150 * ones(num_b, 1)] * 1e-3;
power.P = P_list * ones(1, dim.N);                      % peak power constraints
power.ratio = 0.8;                                      % ratio of the average to the peak
power.P_bar = P_list .* power.ratio;                    % average power constraints
power.L_0 = 1e-3;                                       % channel fading at reference distance (1m)
power.E = gain.delta + gain.sigma + mean(mu.^2, 1);     % expectation of signal power
power.H = 100.0;                                        % UAV hovering altitude
power.w = w;                                            % trajectory of sensors

% UAV mobility constraints
uav.slot = 50.0 / dim.N;            % duration of each time slot (duration / # of time slots)
uav.Vm = 20.0;                      % UAV maximum speed
uav.q_init = [200.0 0.0];           % UAV initial position



% alternating opt initialization

% c_iter -> init c
% c_iter = ones(dim.K, dim.N) * 1e-8;

% q_iter -> init UAV trajectory
centroid = squeeze(mean(power.w));
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



% opt parameter settings
sca.momentum = 0.8;
sca.epsilon = 1e-3;
sca.patience = 10;
% momentum和patience之间应该有关联，momentum大，说明对历史信息的利用率高，patience也应该大
% 因为有momentum的存在，参数更新总是落后于当前求解器找到的最优值
% 所以必须加上patience，让参数再多更新几轮，尽可能追上求解器找到的最优值


% [c_iter, a_iter] = solve_c(q_iter, dim, power, gain, sca, 1);
% q_iter = solve_q(c_iter, dim, power, uav, 1);


epsilon = 1e-3;
gain_iter = 0;
gain_list = [];


% [c_iter, a_iter] = solve_c(q_iter, dim, power, gain, sca, 1);
[c_iter, a_iter] = solve_c_alter(q_iter, dim, power, gain, sca, 1);


% while 1
% 
%     
%     q_iter = solve_q(c_iter, dim, power, uav, 1);
%     [c_iter, a_iter] = solve_c(q_iter, dim, power, gain, sca, 1);
%     
%     gain_opt = sum(a_iter);
%     gain_list = [gain_list gain_opt];
% 
%     if abs(gain_opt - gain_iter) < epsilon
%         break;
%     end
%     gain_iter = gain_opt;
% 
% end


% factor = sum(c_iter, 'all') / sum(sum(c_iter) .^ 2);
% factor = 1 ./ sum(c_iter);
% z_hat = factor .* (squeeze(sum(reshape(c_iter, [1 size(c_iter)]) .* z, 2)) + randn(1, N) * sqrt(delta_0));
% predicted = predict(Mdl, z_hat);
% accuracy = sum(predicted == label_test) / (L * num_test_per_class);
% accuracy_list = [accuracy_list accuracy];


% % plot results
% figure('Position', [450 100 560 600]);
% subplot(211);
% plot(gain_list, '-o', 'MarkerFaceColor', 'b');
% title('discriminant gain');
% xlabel('iteration');
% 
% subplot(212);
% plot(accuracy_list, '-o', 'MarkerFaceColor', 'r');
% title('accuracy');
% xlabel('iteration');
% 
% load('./q.mat');
% figure;
% title('trajectory');
% axis([0 400 0 400]);
% grid on;
% 
% start_a = [50.0 150.0];
% end_a = [50.0 350.0];
% start_b = [350.0 150.0];
% end_b = [250.0 325.0];
% center_a = [linspace(start_a(1), end_a(1), N)' linspace(start_a(2), end_a(2), N)'];
% center_b = [linspace(start_b(1), end_b(1), N)' linspace(start_b(2), end_b(2), N)'];
% 
% hold on;
% plot(center_a(:, 1), center_a(:, 2));
% plot(center_b(:, 1), center_b(:, 2));
% plot(q_iter(:, 1), q_iter(:, 2), 'b');
% legend('cluster A', 'cluster B', 'UAV');
% 
% init_a = w(1: num_a, 1, :);
% fin_a = w(1: num_a, end, :);
% init_b = w(num_a + 1: end, 1, :);
% fin_b = w(num_a + 1: end, end, :);
% 
% scatter(init_a(:, 1), init_a(:, 2), 'HandleVisibility', 'off');
% scatter(init_b(:, 1), init_b(:, 2), 'HandleVisibility', 'off');
% scatter(fin_a(:, 1), fin_a(:, 2), 'HandleVisibility', 'off');
% scatter(fin_b(:, 1), fin_b(:, 2), 'HandleVisibility', 'off');
% 
% % mac截图快捷键：control + cmd + shift + 4
