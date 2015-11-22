%% Project 3 - Classification
%% Nandakishore Krishna
%% Person number : 50169797

clear; close all; clc;

UBitName = 'Nanda Kishore Krishna';
personNumber = '50169797';

format long g

images = loadMNISTImages('../data/train-images.idx3-ubyte');
labels = loadMNISTLabels('../data/train-labels.idx1-ubyte');

% 10 digits
k = 10;

d = 784;

% Logistic regression weights D x K
Wlr = zeros(d, k);

% LR biases 1 x K 
blr = zeros(1, k);
size(blr)

save('proj3.mat');