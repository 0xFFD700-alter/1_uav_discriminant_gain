% clc;
clear;close all;

dir_prefix = '../temp/data/THREE_RADAR_STFT_MAT/';
dir_radar = [dir_prefix 'radar_%d/mat_radar%d_%d.mat'];
num_radar = 2;
num_class = 4;
data = zeros(num_class, 1000, num_radar, 1520);
for radar = 1:num_radar
    for class = 1:num_class
        load(sprintf(dir_radar, radar, radar, class))
        mat = abs(real(mat(:, 1:10:400, :))) + abs(imag(mat(:, 1:10:400, :)));
        mat = reshape(mat, 1000, 1520);
        mean_mat = mean(mat, 2);
        std_mat = max(std(mat, 0, 2), 1 / sqrt(1520));
        data(class, :, radar, :) = reshape((mat - mean_mat) ./ std_mat, 1, 1000, 1, 1520);
    end
end

%%
pca_dim = 12;
num_train = 800;
data_pca = zeros(num_class, 1000, num_radar, pca_dim);
for radar = 1:num_radar
    [coeff, score] = pca(reshape(data(:, 1:num_train, radar, :), num_class * num_train, 1520));
    data_pca(:, 1:num_train, radar, :) = reshape(score(:, 1:pca_dim), num_class, num_train, 1, pca_dim);
    data_pca(:, num_train + 1:1000, radar, :) = reshape(reshape(data(:, num_train + 1:1000, radar, :), num_class * (1000 - num_train), 1520) * coeff(:, 1:pca_dim), num_class, 1000 - num_train, 1, pca_dim);
end

%%
data_pca_normed = (data_pca - mean(data_pca, [2 3])) ./ std(data_pca, 0, [1 2 3]) + mean(data_pca, [2 3]);
mu = squeeze(mean(data_pca_normed, [2 3]));
sigma = reshape(var(data_pca_normed, 0, [1 2 3]), 1, pca_dim);

label_train = reshape(repmat(char((1:num_class) + abs('0'))', [1, num_train * num_radar]), num_class * num_train * num_radar, 1);
label_test = reshape(repmat(char((1:num_class) + abs('0'))', [1, (1000 - num_train) * num_radar]), num_class * (1000 - num_train) * num_radar, 1);

data_train = reshape(data_pca_normed(:, 1:num_train, :, :), num_class * num_train * num_radar, pca_dim);
data_test = reshape(data_pca_normed(:, num_train + 1:1000, :, :), num_class * (1000 - num_train) * num_radar, pca_dim);

Mdl = fitcecoc(data_train, label_train);
label_pred = predict(Mdl, data_test);
accuracy = sum(label_pred == label_test) / (num_class * (1000 - num_train) * num_radar);
fprintf('accuracy: %f\n', accuracy);

x = data_test;
save('./data/inference/mean.mat', 'mu');
save('./data/inference/var.mat', 'sigma');
save('./data/inference/data_test.mat', 'x');
save('./data/inference/label_test.mat', 'label_test')
save('./Mdl.mat', 'Mdl');
