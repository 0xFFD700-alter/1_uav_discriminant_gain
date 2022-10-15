% clc;
clear;close all;
load('./data/inference/mean.mat'); % mean value
load('./data/inference/var.mat'); % ground-true var
load('./data/inference/var_sensor.mat'); % distortion var
load('./data/inference/trajectory_sensor.mat'); % trajectory of sensors
load('./data/inference/data_test_noise.mat'); % distorted data
load('./data/inference/label_test.mat'); % label of test data
load('./Mdl.mat'); % SVM model 12-dim



% data and model to perform inference tasks
eval.L = size(mu, 1);
eval.num_test_samples = size(z, 1);
eval.z = z;
eval.Mdl = Mdl;
eval.label_test = label_test;



% parameter settings
rng(2022);

% data dimensions
dim.N = size(mu, 2);                % N -> # of feature dims (# of time slots)
dim.K = size(delta, 1);             % K -> # of sensors
num_a = fix(dim.K * 0.3) + 1;
num_b = dim.K - num_a;

% parameters of data distribution (used to compute discriminant gain)
gain.u = zeros(1, dim.N);
for i = 1: eval.L - 1
    for j = i + 1: eval.L
        gain.u = gain.u + (mu(i, :) - mu(j, :)) .^ 2;
    end
end
gain.u = gain.u * 2 / eval.L / (eval.L - 1);          % average square mean
gain.sigma = sigma;                         % variance of ground truth
gain.delta = delta;                         % variance of distortion
gain.delta_0 = 1e-11;                       % variance of Gaussian noise

% power constraints
P_list = [8 * ones(num_a, 1); 8 * ones(num_b, 1)] * 1e-3;
power.P = P_list * ones(1, dim.N);                      % peak power constraints
power.ratio = 0.8;                                      % ratio of the average to the peak
power.P_bar = P_list .* power.ratio;                    % average power constraints
power.L_0 = 1e-4;                                       % channel fading at reference distance (1m)
power.E = gain.delta + gain.sigma + mean(mu.^2, 1);     % expectation of signal power
power.H = 100.0;                                        % UAV hovering altitude
power.w = w;                                            % trajectory of sensors

% UAV mobility constraints
uav.slot = 50.0 / dim.N;            % duration of each time slot (duration / # of time slots)
uav.Vm = 20.0;                      % UAV maximum speed
uav.q_init = [200.0 0.0];           % UAV initial position



% alternating opt initialization

% c_iter -> init c
c_iter = ones(dim.K, dim.N) * 1e-6;

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



% sca opt parameter settings
sca.momentum = 0.8;
sca.epsilon = 1e-3;
sca.patience = 5;
% momentum和patience之间应该有关联，momentum大，说明对历史信息的利用率高，patience也应该大
% 因为有momentum的存在，参数更新总是落后于当前求解器找到的最优参�?
% �?以必须加上patience，让参数再多更新几轮，尽可能追上求解器找到的�?优参�?

% opt parameter settings
epsilon = 1e-3;
patience = 5;

% auxiliary variables for alternating opt
gain_iter = 0;
gain_list = [];
accuracy_list = [];
gain_fun = @(x) sum(sum(x) .^ 2 .* gain.u ./ (gain.sigma .* sum(x) .^ 2 + sum(x .^ 2 .* gain.delta) + gain.delta_0));
patience_count = 0;
repeat = 20;

q_iter = solve_q(c_iter, dim, power, uav, 1);

% while 1
%     c_iter = solve_c_alter(q_iter, dim, power, gain, sca, 1);
%     
%     gain_opt = gain_fun(c_iter);
%     gain_list = [gain_list gain_opt];
%     accuracy = inference(c_iter, dim, gain, eval, repeat);
%     accuracy_list = [accuracy_list accuracy];
% 
%     fprintf('\naccuracy: %f, gain_opt: %f\n', accuracy, gain_opt);
% 
%     if abs(gain_opt - gain_iter) <= epsilon
%         patience_count = patience_count + 1;
%         if patience_count > patience
%             break
%         end
%     else
%         patience_count = 0;
%     end
% 
%     gain_iter = gain_opt;
%     q_iter = solve_q(c_iter, dim, power, uav, 1);
% end


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
