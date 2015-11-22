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

d = size(images,1);

% target matrix, label 0 is mapped to 1, label 1 to 2 and so on
T = zeros(k, length(images));
for i = 1 : k
    T(i, :) = (labels == (i-1));
end

% Logistic regression weights D x K
Wlr = zeros(d, k);

% LR biases 1 x K 
blr = 0.01 * ones(1, k);

% learning rate
eta = 1;

% error
lgr_error = zeros(1, length(images));

% gradient descent
for i = 1 : length(images)
    ak = Wlr' * images(:, i) + blr';
    
    y = zeros(k, 1);
    exp_ak = exp(ak);
    sigma_ak = sum(exp(ak));
    for j = 1 : k
        y(j, 1) = exp_ak(j, 1) / sigma_ak;
    end
    Wlr = Wlr - eta * ( images(:, i) * (y - T(:, i))');
    lgr_error(1, i) = -1 * sum(T(:, i) - y);
end

plot(1:length(images), lgr_error);



%save('proj3.mat');