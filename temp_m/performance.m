%% fig: discriminant gain convergence
clc;clear;close all;
load('./data/inference/data_test.mat', 'x');
K = 5;
N = size(x, 2);
gen_distorted_data(K, x);
gen_sensor_trajectory(K, N);
gen_parameter;
model_1 = ["./data/model/svm_classifier_12dim.mat", "svm_classifier_12dim"];
model_2 = ["./data/model/mlp_classifier_12dim.mat", "mlp_classifier_12dim"];
p_dbm = 6;
peak_p = [10^(p_dbm / 10) 10^(p_dbm / 10)];
[gain_list, acc_list_1, acc_list_2] = alt_main_gain_acc(peak_p, model_1, model_2);

%%
start_idx = 1;
end_idx = 25;
plt1 = smooth(gain_list(start_idx:end_idx), 12);
plt2 = smooth(acc_list_1(start_idx:end_idx) * 100, 12);
plt3 = smooth(acc_list_2(start_idx:end_idx) * 100, 12);

figure % left: gain, right: acc
grid on
yyaxis left
plot(plt1(1:23), '-', 'linewidth', 1.4, 'color', [0 0 0.4]);
ylim([5.05 5.75]);
ylabel('Discriminant gain $G$', 'interpreter', 'latex');
set(gca, 'YColor', [0 0 0.4]);
yyaxis right
plot(plt2(1:23), '--', 'linewidth', 1.4, 'color', [0 0.4 0.4]);
hold on
plot(plt3(1:23), ':', 'linewidth', 1.4, 'color', [0 0.4 0.4]);
ylabel('Inference accuracy ($\%$)', 'interpreter', 'latex');
set(gca, 'YColor', [0 0.4 0.4]);
legend('Discriminant gain', 'SVM accuracy', 'MLP accuracy', 'location', 'southeast', 'fontsize', 10, 'box', 'on', 'edgecolor', [0.7 0.7 0.7], 'linewidth', 0.3);
set(gca, 'GridLineStyle', '-', 'GridAlpha', 0.15);
xlabel('Iterations', 'interpreter', 'latex');

%% fig: accuracy vs. discriminant gain
clc;clear;close all;
load('./data/inference/data_test.mat', 'x');
K = 5;
N = size(x, 2);
gen_distorted_data(K, x);
gen_sensor_trajectory(K, N);
gen_parameter;

model_1 = ["./data/model/svm_classifier_12dim.mat", "svm_classifier_12dim"];
model_2 = ["./data/model/mlp_classifier_12dim.mat", "mlp_classifier_12dim"];
gain_list_cell = {};
acc_list_1_cell = {};
acc_list_2_cell = {};
p_dbm_list = [3 9];
for p_dbm = p_dbm_list
    peak_p = [10^(p_dbm / 10) 10^(p_dbm / 10)];
    [gain_list, acc_list_1, acc_list_2] = alt_main_gain_acc(peak_p, model_1, model_2);
    gain_list_cell = [gain_list_cell, gain_list];
    acc_list_1_cell = [acc_list_1_cell, acc_list_1];
    acc_list_2_cell = [acc_list_2_cell, acc_list_2];
end

%%
gain_list_plot = cell2mat(gain_list_cell);
acc_list_1_plot = cell2mat(acc_list_1_cell);
acc_list_2_plot = cell2mat(acc_list_2_cell);
[gain_list_plot, idx] = sort(gain_list_plot);
acc_list_1_plot = acc_list_1_plot(idx);
acc_list_2_plot = acc_list_2_plot(idx);

start_idx = 1;

figure % acc_1
plot(gain_list_plot(start_idx:end), acc_list_1_plot(start_idx:end) * 100, '-', 'linewidth', 1.6, 'color', [0 0 0.5]);
set(gca, 'GridLineStyle', '-', 'GridAlpha', 0.15);
hold on 
grid on
plot(gain_list_plot(start_idx:end), acc_list_2_plot(start_idx:end) * 100, '-.', 'linewidth', 1.6, 'color', [0 0.5 0.5]);
legend('SVM', 'MLP', 'location', 'southeast', 'fontsize', 10, 'box', 'on', 'edgecolor', [0.7 0.7 0.7], 'linewidth', 0.3);
xlabel('Discriminant gain $G$', 'interpreter', 'latex')
ylabel('Inference accuracy ($\%$)', 'interpreter', 'latex');

acc_list_1_plot = smooth(gain_list_plot, acc_list_1_plot, 20);
acc_list_2_plot = smooth(gain_list_plot, acc_list_2_plot, 20);

figure % acc_1
plot(gain_list_plot(start_idx:end), acc_list_1_plot(start_idx:end) * 100, '-', 'linewidth', 1.6, 'color', [0 0 0.5]);
set(gca, 'GridLineStyle', '-', 'GridAlpha', 0.15);
hold on 
grid on
plot(gain_list_plot(start_idx:end), acc_list_2_plot(start_idx:end) * 100, '-.', 'linewidth', 1.6, 'color', [0 0.5 0.5]);
legend('SVM', 'MLP', 'location', 'southeast', 'fontsize', 10, 'box', 'on', 'edgecolor', [0.7 0.7 0.7], 'linewidth', 0.3);
xlabel('Discriminant gain $G$', 'interpreter', 'latex')
ylabel('Inference accuracy ($\%$)', 'interpreter', 'latex');

%% fig: number of sensors
clc;clear;close all;
load('./data/inference/data_test.mat', 'x');
N = size(x, 2);
peak_p = [10^0.2 10^0.2];

model_1 = ["./data/model/svm_classifier_12dim.mat", "svm_classifier_12dim"];
model_2 = ["./data/model/mlp_classifier_12dim.mat", "mlp_classifier_12dim"];
K_list = 4:1:9;
acc_K_plot_1 = zeros(3, length(K_list));
acc_K_plot_2 = zeros(3, length(K_list));
for ii = 1:length(K_list)
    K = K_list(ii);
    gen_distorted_data(K, x);
    gen_sensor_trajectory(K, N);
    gen_parameter;
    [acc_list_1, acc_list_2] = alt_main(peak_p, model_1, model_2);
    acc_K_plot_1(1, ii) = acc_list_1(end);
    acc_K_plot_2(1, ii) = acc_list_2(end);
    acc_K_plot_1(2, ii) = acc_list_1(1);
    acc_K_plot_2(2, ii) = acc_list_2(1);
    fixq = repmat([200 200], [size(q_iter, 1) 1]);
    [acc_1, acc_2] = fixq_main(peak_p, model_1, model_2, fixq);
    acc_K_plot_1(3, ii) = acc_1;
    acc_K_plot_2(3, ii) = acc_2;
end

end_idx = length(K_list);

figure % acc_1
plot(K_list(1:end_idx), acc_K_plot_1(1, 1:end_idx) * 100, '^-', 'linewidth', 1.6, 'color', [0 0 1]);
set(gca, 'GridLineStyle', '-', 'GridAlpha', 0.15, 'xtick', K_list(1:end_idx));
hold on 
grid on
plot(K_list(1:end_idx), acc_K_plot_1(2, 1:end_idx) * 100, 's--', 'linewidth', 1.6, 'color', [0.75 0 0.75]);
plot(K_list(1:end_idx), acc_K_plot_1(3, 1:end_idx) * 100, 'o-.', 'linewidth', 1.6, 'color', [1 0 0]);
legend('Joint Optimization (proposed)', 'Fly-hover', 'Baseline', 'location', 'southeast', 'fontsize', 10, 'box', 'on', 'edgecolor', [0.7 0.7 0.7], 'linewidth', 0.3);
xlabel('Number of sensors $N$', 'interpreter', 'latex')
ylabel('Inference accuracy ($\%$)', 'interpreter', 'latex');
ylim([30 100]);

figure; % acc_2
plot(K_list(1:end_idx), acc_K_plot_2(1, 1:end_idx) * 100, '^-', 'linewidth', 1.6, 'color', [0 0 1]);
set(gca, 'GridLineStyle', '-', 'GridAlpha', 0.15, 'xtick', K_list(1:end_idx));
hold on 
grid on
plot(K_list(1:end_idx), acc_K_plot_2(2, 1:end_idx) * 100, 's--', 'linewidth', 1.6, 'color', [0.75 0 0.75]);
plot(K_list(1:end_idx), acc_K_plot_2(3, 1:end_idx) * 100, 'o-.', 'linewidth', 1.6, 'color', [1 0 0]);
legend('Joint Optimization (proposed)', 'Fly-hover', 'Baseline', 'location', 'southeast', 'fontsize', 10, 'box', 'on', 'edgecolor', [0.7 0.7 0.7], 'linewidth', 0.3);
xlabel('Number of sensors $N$', 'interpreter', 'latex')
ylabel('Inference accuracy ($\%$)', 'interpreter', 'latex');
ylim([30 100]);


%% fig: peak power change
clc;clear;close all;
load('./data/inference/data_test.mat', 'x');
K = 5;
N = size(x, 2);
gen_distorted_data(K, x);
gen_sensor_trajectory(K, N);
gen_parameter;

model_1 = ["./data/model/svm_classifier_12dim.mat", "svm_classifier_12dim"];
model_2 = ["./data/model/mlp_classifier_12dim.mat", "mlp_classifier_12dim"];
p_dbm = 2:1:10;
acc_power_plot_1 = zeros(3, length(p_dbm));
acc_power_plot_2 = zeros(3, length(p_dbm));
for ii = 1:length(p_dbm)
    peak_p = [10^(p_dbm(ii) / 10) 10^(p_dbm(ii) / 10)];
    [acc_list_1, acc_list_2] = alt_main(peak_p, model_1, model_2);
    acc_power_plot_1(1, ii) = acc_list_1(end);
    acc_power_plot_2(1, ii) = acc_list_2(end);
    acc_power_plot_1(2, ii) = acc_list_1(1);
    acc_power_plot_2(2, ii) = acc_list_2(1);
    fixq = repmat([250 250], [size(q_iter, 1) 1]);
    [acc_1, acc_2] = fixq_main(peak_p, model_1, model_2, fixq);
    acc_power_plot_1(3, ii) = acc_1;
    acc_power_plot_2(3, ii) = acc_2;
end

end_idx = length(p_dbm);

figure % acc_1
plot(p_dbm(1:end_idx), acc_power_plot_1(1, 1:end_idx) * 100, '^-', 'linewidth', 1.6, 'color', [0 0 1]);
hold on
grid on
plot(p_dbm(1:end_idx), acc_power_plot_1(2, 1:end_idx) * 100, 's--', 'linewidth', 1.6, 'color', [0.75 0 0.75]);
plot(p_dbm(1:end_idx), acc_power_plot_1(3, 1:end_idx) * 100, 'o-.', 'linewidth', 1.6, 'color', [1 0 0]);
lgd = legend('Joint Optimization (proposed)', 'Fly-hover', 'Baseline', 'location', 'southeast', 'fontsize', 10, 'box', 'on', 'edgecolor', [0.7 0.7 0.7], 'linewidth', 0.3);
% lgd.FontName = 'Candara';
xlabel('Peak transmit power $P_{k}$ (dbm)', 'interpreter', 'latex')
ylabel('Inference accuracy ($\%$)', 'interpreter', 'latex');
set(gca, 'GridLineStyle', '-', 'GridAlpha', 0.15, 'xtick', p_dbm(1:end_idx));
yticks([20 30 40 50 60 70 80 90 100]);
ylim([20 100]);

figure; % acc_2
plot(p_dbm(1:end_idx), acc_power_plot_2(1, 1:end_idx) * 100, '^-', 'linewidth', 1.6, 'color', [0 0 1]);
hold on
grid on
plot(p_dbm(1:end_idx), acc_power_plot_2(2, 1:end_idx) * 100, 's--', 'linewidth', 1.6, 'color', [0.75 0 0.75]);
plot(p_dbm(1:end_idx), acc_power_plot_2(3, 1:end_idx) * 100, 'o-.', 'linewidth', 1.6, 'color', [1 0 0]);
lgd = legend('Joint Optimization (proposed)', 'Fly-hover', 'Baseline', 'location', 'southeast', 'fontsize', 10, 'box', 'on', 'edgecolor', [0.7 0.7 0.7], 'linewidth', 0.3);
% lgd.FontName = 'Candara';
xlabel('Peak transmit power $P_{k}$ (dbm)', 'interpreter', 'latex')
ylabel('Inference accuracy ($\%$)', 'interpreter', 'latex');
set(gca, 'GridLineStyle', '-', 'GridAlpha', 0.15, 'xtick', p_dbm(1:end_idx));
yticks([20 30 40 50 60 70 80 90 100]);
ylim([20 100]);


%% fig: uav trajectory
clc;clear;close all;
load('./data/inference/data_test.mat', 'x');
K = 8;
N = size(x, 2);
gen_distorted_data(K, x);
gen_sensor_trajectory(K, N);
gen_parameter;

model_1 = ["./data/model/svm_classifier_12dim.mat", "svm_classifier_12dim"];
model_2 = ["./data/model/mlp_classifier_12dim.mat", "mlp_classifier_12dim"];
p_dbm = 10;
peak_p = [10^(p_dbm / 10) 10^(p_dbm / 10)];
hover_q = q_iter;
q_iter = alt_main_uav(peak_p, model_1, model_2);
fix_q = [250 250];

start_a = [50.0 50.0];
start_b = [450.0 50.0];
load('./data/inference/trajectory_sensor_centroid_a.mat', 'centroid_a');
load('./data/inference/trajectory_sensor_centroid_b.mat', 'centroid_b');

%%
figure;
t1 = plot(q_iter(:, 1), q_iter(:, 2), '--<', 'LineWidth', 1.6, 'MarkerSize', 6, 'color', [0.6 0 0.6]);
hold on
grid on
t2 = plot(hover_q(:, 1), hover_q(:, 2), '-.d', 'LineWidth', 1.6, 'MarkerSize', 6, 'color', [0.8 0.3 0]);
t3 = plot(fix_q(1), fix_q(2), 'o', 'LineWidth', 1.6, 'MarkerSize', 7, 'color', [0 0 0]);
t4 = plot(centroid_a(:, 1), centroid_a(:, 2), '-^', 'LineWidth', 1.2, 'color', [0 0 0.6]);
t5 = plot(centroid_b(:, 1), centroid_b(:, 2), '-^', 'LineWidth', 1.2, 'color', [0 0 0.6]);
init_a = w(1: num_a, 1, :);
fin_a = w(1: num_a, end, :);
init_b = w(num_a + 1: end, 1, :);
fin_b = w(num_a + 1: end, end, :);
scatter(init_a(:, 1), init_a(:, 2), 'ks', 'HandleVisibility', 'off', 'LineWidth', 1.2);
scatter(init_b(:, 1), init_b(:, 2), 'ks', 'HandleVisibility', 'off', 'LineWidth', 1.2);
scatter(fin_a(:, 1), fin_a(:, 2), 'ks', 'HandleVisibility', 'off', 'LineWidth', 1.2);
scatter(fin_b(:, 1), fin_b(:, 2), 'ks', 'HandleVisibility', 'off', 'LineWidth', 1.2);
xlabel('$x(\mathrm{m})$', 'interpreter', 'latex')
ylabel('$y(\mathrm{m})$', 'interpreter', 'latex');
set(gca, 'GridLineStyle', '-', 'GridAlpha', 0.15);
xticks(0:50:500);
xlim([0 500]);
yticks(0:50:500);
ylim([0 500]);
legend([t1 t2 t3], 'Joint Optimization', 'Fly-hover', 'Baseline', 'location', 'southeast', 'fontsize', 10, 'box', 'on', 'edgecolor', [0.7 0.7 0.7], 'linewidth', 0.3);
a1 = annotation('textarrow', [0.285, 0.285], [0.39, 0.65]);
a2 = annotation('textarrow', [0.84, 0.75], [0.57, 0.81]);
set(a1, 'String', 'Cluster A', 'FontSize', 10, 'HeadWidth', 6);
set(a2, 'String', 'Cluster B', 'FontSize', 10, 'HeadWidth', 6);

%%
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