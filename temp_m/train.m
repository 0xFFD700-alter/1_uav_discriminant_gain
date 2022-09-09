clc;clear;close all;
% load fisheriris
% X = meas;
% Y = species;

load('./data/training/data.mat')
load('./data/training/label.mat')
num_class = size(label, 1);
num_per_class = size(label, 2);
num_train_per_class = 800;
num_test_per_class = 200;

data_train = data(:, 1: num_train_per_class, :);
data_test = data(:, num_train_per_class + 1: end, :);
label_train = label(:, 1: num_train_per_class);
label_test = label(:, num_train_per_class + 1: end);
label = reshape(label, [size(label, 1) * size(label, 2) 1]);
% label_cell = cell();
% for i = 1: size(label, 1)
%     for j = 1: size(label, 2)
%         label = {label}
% label = {label};