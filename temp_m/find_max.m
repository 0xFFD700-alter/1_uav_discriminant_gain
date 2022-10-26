% clc;
clear;close all;

load('./data/inference/mean.mat'); % mean value
load('./data/inference/var.mat'); % ground-true var
load('./data/inference/var_sensor.mat'); % distortion var

% mu = mu(:, 1:2);
% delta = delta(1:3, 1:2);
% sigma = sigma(:, 1:2);

L = size(mu, 1);
N = size(mu, 2);
K = size(delta, 1);

delta_0 = 0;

u = zeros(1, N);
for l1 = 1: L - 1
    for l2 = l1 + 1: L
        u = u + (mu(l1, :) - mu(l2, :)) .^ 2;
    end
end
u = u * 2 / L / (L - 1);
fun = @(x) sum(sum(reshape(x, [K N])) .^ 2 .* u ...
./ (sigma .* sum(reshape(x, [K N])) .^ 2 + sum(reshape(x, [K N]) .^ 2 .* delta) + delta_0));

nvars = K * N;
[c, fval, ~, ~] = particleswarm(fun, nvars);

%%
% clear;close all;
load('./data/inference/mean.mat'); % mean value
load('./data/inference/var.mat'); % ground-true var
load('./data/inference/var_sensor.mat'); % distortion var

L = size(mu, 1);
N = size(mu, 2);
K = size(delta, 1);

delta_0 = 0;
sca_momentum = 0.8;

c_iter = ones(K, N) * 1e-3;
a_iter = ones(1, N) * 1e-3;

u = zeros(1, N);
for l1 = 1: L - 1
    for l2 = l1 + 1: L
        u = u + (mu(l1, :) - mu(l2, :)) .^ 2;
    end
end
u = u * 2 / L / (L - 1);

epsilon = 1e-3;
gain_opt = 0;
% gain_list = sum(a_iter);

while 1
    cvx_begin
    cvx_solver mosek
    variable c(K, N) nonnegative
    variable a(1, N) nonnegative
    maximize sum(a)

    subject to
        u ./ a_iter .* sum(c_iter) .^ 2 ...
        + sum(c - c_iter) * 2 .* u ./ a_iter .* sum(c_iter) ...
        - (a - a_iter) .* u ./ a_iter .^ 2 .* sum(c_iter) .^ 2 ...
        - sum(c) .^ 2 .* sigma ...
        >= sum(c .^ 2 .* delta) + delta_0;
    cvx_end

    assert(strcmp(cvx_status, 'Solved'));
    gain_iter = gain_opt;
    gain_opt = cvx_optval;

    c_iter = sca_momentum * c_iter + (1 - sca_momentum) * c;
    a_iter = sca_momentum * a_iter + (1 - sca_momentum) * a;

    if abs(gain_opt - gain_iter) < epsilon
        break;
    end
end