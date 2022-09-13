clc;clear;close all;

load('./data/training/data.mat')
load('./data/training/label.mat')
pca_dim = size(data, 3);
num_class = size(label, 1);
num_per_class = size(label, 2);
num_train_per_class = 800;
num_test_per_class = 200;

% X = reshape(data, [num_class * num_per_class pca_dim]);
% Y = reshape(label, [num_class * num_per_class 1]);
X_train = reshape(data(:, 1: num_train_per_class, :), [num_class * num_train_per_class, pca_dim]);
X_test = reshape(data(:, num_train_per_class + 1: end, :), [num_class * num_test_per_class, pca_dim]);
Y_train = reshape(label(:, 1: num_train_per_class), [num_class * num_train_per_class 1]);
Y_test = reshape(label(:, num_train_per_class + 1: end), [num_class * num_test_per_class 1]);

Mdl = fitcecoc(X_train, Y_train);
Y_pred = predict(Mdl, X_test);
accuracy = sum(Y_pred == Y_test) / (num_class * num_test_per_class);

save('./Mdl.mat', 'Mdl');
