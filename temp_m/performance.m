%% fig: accuracy vs. discriminant gain
clc;clear;close all;
p_list = [7.5 8:1:60];
dis_vs_plot = [];
acc_vs_plot = [];
for p = p_list
    str_power = sprintf('%0.1f', p);
    load(['./data/results/accuracy_list_', str_power, '.mat']);
    load(['./data/results/gain_list_', str_power, '.mat']);
%     acc_plot = [acc_plot accuracy_list];
%     dis_plot = [dis_plot gain_list];
    fprintf('%f  %f\n', accuracy_list(1), accuracy_list(4));
end

% figure;
% plot(acc_vs_plot); hold on;
% plot(dis_vs_plot);


%% fig: peak power change
clc;clear;close all;
load('./data/inference/data_test.mat', 'x');
K = 5;
N = size(x, 2);
gen_distorted_data(K, x);
gen_sensor_trajectory(K, N);
clear;
gen_parameter;

% 1e-10: [8,64]
% delta 8: 6.4 dis: 6.60
% delta 16: 7.5 dis: 5.84
model = ["./data/model/svm_classifier_12dim.mat", "svm_classifier_12dim"];
p_list = 1:1:9; %[5.8:0.5:10 10:5:30];%[8.5:0.5:9.5 10:5:40];
dis_power_plot = zeros(3, length(p_list));
acc_power_plot = zeros(3, length(p_list));
for i = 1:length(p_list)
    peak_p = [p_list(i) p_list(i)];
    [dis_list, acc_list] = alt_main(peak_p, model);
    dis_power_plot(1, i) = dis_list(end);
    acc_power_plot(1, i) = acc_list(end);
    dis_power_plot(2, i) = dis_list(1);
    acc_power_plot(2, i) = acc_list(1);
    fixq = repmat(mean(q_iter), [size(q_iter, 1) 1]);
    [dis, acc] = fixq_main(peak_p, model, fixq);
    dis_power_plot(3, i) = dis;
    acc_power_plot(3, i) = acc;
end

% figure; % discriminant gain
% grid on;
% plot(p_list(1:7), dis_power_plot(1, 1:7)); hold on;
% plot(p_list(1:7), dis_power_plot(2, 1:7));
% % plot(p_list, dis_power_plot(3, :));
% % legend('alt', 'fly-hover', 'fix');
% legend('alt', 'fly-hover','Location','SouthEast');
% 
% figure; % accuracy
% grid on;
% plot(p_list(1:7), acc_power_plot(1, 1:7)); hold on;
% plot(p_list(1:7), acc_power_plot(2, 1:7));
% % plot(p_list, acc_power_plot(3, :));
% % legend('alt', 'fly-hover', 'fix');
% legend('alt', 'fly-hover','Location','SouthEast');

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