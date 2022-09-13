clc;clear;close all;
num_train_per_class = 800;
num_test_per_class = 200;
load('./data/inference/mean.mat'); % mean value
load('./data/inference/var.mat'); % ground-true var
load('./data/inference/var_sensor.mat'); % distortion var
load('./data/inference/trajectory_sensor.mat'); % trajectory of sensors
load('./data/inference/data_test_noise.mat'); % distorted data
load('./data/inference/label_test.mat'); % label of test data

L = size(mu, 1);
N = size(mu, 2);
K = size(z, 2);

num_a = fix(K * 0.3) + 1;
num_b = K - num_a;
P_list = [0.05 * ones(num_a, 1); 0.03 * ones(num_b, 1)];
P_list = P_list * 300;

H = 100.0;                                      % m
T = 30.0;                                       % s
slot = T / N;                                   % s
Vm = 20.0;                                      % m/s
q_init = [200.0 0.0];                           % m
delta_0 = 1e-11;                                % W
P = P_list * ones(1, N);                        % W
P_bar = P_list ./ 2;                            % W
L_0 = 1e-4;                                     % dB
sca_momentum = 1 - 1e-1;

E = delta + sigma + mean(mu.^2, 1);
c_iter = sqrt(P .* L_0 ./ E ./ 6.6e5);

u = zeros(1, N);
for l1 = 1: L - 1
    for l2 = l1 + 1: L
        u = u + (mu(l1, :) - mu(l2, :)) .^ 2;
    end
end
u = u * 2 / L / (L - 1);

a_iter = sum(c_iter, 1) .^ 2 .* u ./ (sum(c_iter, 1) .^ 2 .* sigma + sum(delta .* c_iter .^ 2, 1) + delta_0) ./ 1e1;

epsilon = 1e-3;
epsilon_sca = 1;
gain = 0;
gain_list = sum(a_iter);

%%
load('./Mdl.mat');
accuracy_list = [];
z_hat = squeeze(sum(reshape(c_iter, [1 size(c_iter)]) .* z, 2)) + randn(1, N) * sqrt(delta_0);
predicted = predict(Mdl, z_hat);
accuracy = sum(predicted == label_test) / (L * num_test_per_class);
accuracy_list = [accuracy_list accuracy];

while 1
    cvx_begin
    cvx_solver mosek
    variable q(N, 2)
    expression vstack_1(K, N)
    expression q_norm(N, 1)
    for k = 1: K
        vstack_1(k, :) = sum((q - squeeze(w(k, :, :))) .^ 2, 2);
    end
    minimize sum(sum(vstack_1 .* c_iter .^ 2 .* E))

    subject to
    (vstack_1 + H ^ 2) .* c_iter .^ 2 .* E <= P * L_0; 
    sum((vstack_1 + H ^ 2) .* c_iter .^ 2 .* E, 2) <= P_bar * L_0 * N;
    q_norm(1) = norm(q(1, :) - q_init);
    for n = 2: N
        q_norm(n) = norm(q(n, :) - q(n - 1, :));
    end
    q_norm <= slot * Vm * ones(N, 1);
    cvx_end
    
    assert(strcmp(cvx_status, 'Solved'))
    q_iter = q;
    
    while 1
        cvx_begin
        cvx_solver mosek
        variable c(K, N) nonnegative
        variable a(1, N) nonnegative
        maximize sum(a)

        subject to
        vstack_2 = zeros(K, N);
        for k = 1: K
            vstack_2(k, :) = sum((q_iter - squeeze(w(k, :, :))) .^ 2, 2);
        end
        c .^ 2 .* (vstack_2 + H ^ 2) .* E <= P * L_0;
        sum(c .^ 2 .* (vstack_2 + H ^ 2) .* E, 2) <= P_bar * L_0 * N;
        u ./ a_iter .* sum(c_iter) .^ 2 ...
        + sum(c - c_iter) * 2 .* u ./ a_iter .* sum(c_iter) ...
        - (a - a_iter) .* u ./ a_iter .^ 2 .* sum(c_iter) .^ 2 ...
        - sum(c) .^ 2 .* sigma ...
        >= sum(c .^ 2 .* delta) + delta_0;
        cvx_end
        
        assert(strcmp(cvx_status, 'Solved'))
        gain_iter = gain;
        gain = cvx_optval;
        c_iter = sca_momentum * c_iter + (1 - sca_momentum) * c;
        a_iter = sca_momentum * a_iter + (1 - sca_momentum) * a;
        
        if abs(gain - gain_iter) < epsilon_sca
            break
        end
    end
    
    gain_list = [gain_list gain];
    z_hat = squeeze(sum(reshape(c_iter, [1 size(c_iter)]) .* z, 2)) + randn(1, N) * sqrt(delta_0);
    predicted = predict(Mdl, z_hat);
    accuracy = sum(predicted == label_test) / (L * num_test_per_class);
    accuracy_list = [accuracy_list accuracy];

    if abs(gain - gain_iter) < epsilon
        break
    end
end

%%
figure('Position', [450 100 560 600]);
subplot(211);
plot(gain_list, '-o', 'MarkerFaceColor', 'b');
title('discriminant gain');
xlabel('iteration');

subplot(212);
plot(accuracy_list, '-o', 'MarkerFaceColor', 'r');
title('accuracy');
xlabel('iteration');