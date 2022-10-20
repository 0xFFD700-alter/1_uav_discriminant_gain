% clc;
clear;close all;
load('./data/inference/data_test.mat')

K = 5;
N = size(x, 2);
samples = size(x, 1);

% add noise
rng(3407);
delta = 16 * rand([K N]);
z = reshape(x, [samples, 1, N]) + reshape(sqrt(delta), [1 K N]) .* randn([1 K N]);
save('./data/inference/data_test_noise.mat', 'z');
save('./data/inference/var_sensor.mat', 'delta');

% generate trajectory
ratio = 0.3;
num_a = floor(K * 0.3) + 1;
num_b = K - num_a;
radius = 15;
start_a = [50.0 100.0]; end_a = [50.0, 350.0]; % TODO: not staright trajectory
start_b = [350.0 150.0]; end_b = [250.0 325.0]; % TODO: not staright trajectory

% random init position and reshape
rad_a = 2 * pi * rand([num_a 1]);
dist_a = radius * rand([num_a 1]);
rad_b = 2 * pi * rand([num_b 1]);
dist_b = radius * rand([num_b 1]);
init_a = start_a + [dist_a .* cos(rad_a) dist_a .* sin(rad_a)];
init_b = start_b + [dist_b .* cos(rad_b) dist_b .* sin(rad_b)];
fin_a = init_a + reshape(end_a - start_a, [1 2]);
fin_b = init_b + reshape(end_b - start_b, [1 2]);

% interpolate
w_a = zeros([num_a N 2]);
w_b = zeros([num_b N 2]);
for i = 1:num_a
    w_a(i, :, 1) = linspace(init_a(i, 1), fin_a(i, 1), N);
    w_a(i, :, 2) = linspace(init_a(i, 2), fin_a(i, 2), N);
end
for i = 1:num_b
    w_b(i, :, 1) = linspace(init_b(i, 1), fin_b(i, 1), N);
    w_b(i, :, 2) = linspace(init_b(i, 2), fin_b(i, 2), N);
end

w = [w_a; w_b];
save('./data/inference/trajectory_sensor.mat', 'w');
