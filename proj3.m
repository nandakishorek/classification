%% Project 3 - Classification
%% Nandakishore Krishna
%% Person number : 50169797

clear; close all; clc;

UBitName = 'Nanda Kishore Krishna';
personNumber = '50169797';

format long g

% 10 digits
k = 10;

% training set
images = loadMNISTImages('../data/train-images.idx3-ubyte');
labels = loadMNISTLabels('../data/train-labels.idx1-ubyte');

% target matrix, label 0 is mapped to 1, label 1 to 2 and so on
T = zeros(k, length(labels));
for i = 1 : k
    T(i, :) = (labels == (i-1));
end

% validation set
valImages = loadMNISTImages('../data/t10k-images.idx3-ubyte');
valLabels = loadMNISTLabels('../data/t10k-labels.idx1-ubyte');

% target matrix, label 0 is mapped to 1, label 1 to 2 and so on
valT = zeros(k, length(valLabels));
for i = 1 : k
    valT(i, :) = (valLabels == (i-1));
end

d = size(images,1);

% Logistic regression weights D x K
Wlr = zeros(d, k);

% LR biases 1 x K 
blr = 0.1 * ones(1, k);

% learning rate
eta = 1;

% error
lgr_error = zeros(1, length(images));

% gradient descent
for i = 1 : length(images)
    a = Wlr' * images(:, i) + blr';
    
    % normalize a to avoid huge values in softmax
    a = a / max(a);
    
    y = zeros(k, 1);
    exp_a = exp(a);
    sigma_a = sum(exp(a));
    for m = 1 : k
        y(m, 1) = exp_a(m, 1) / sigma_a;
    end
    Wlr = Wlr - eta * ( images(:, i) * (y - T(:, i))' );
    lgr_error(1, i) = -1 * sum(T(:, i) - log(y));
end

% plot(1:length(images), lgr_error);

% validate the weights
predictLGR = bsxfun(@plus, Wlr' * valImages, blr');
[~, c] = max(predictLGR, [], 1); 
c = (c - 1)';

valError = sum(c ~= valLabels) / size(valLabels, 1);

% Neural net
% number of hidden units
j = 5;

% weights for first layer d x j
Wnn1 = zeros(d, j);

% weights for first layer j x k
Wnn2 = zeros(j, k);

% bias for first layer 1 x j
bnn1 = 0.01 * ones(1, j);

% bias for first layer 1 x k
bnn2 = 0.01 * ones(1, k);

% activation function
h = 'sigmoid';

% NN gradient descent
etaNN = 0.1;
for i = 1 : length(images)
    % feed forward propagation
    z = zeros(j, 1);
    for m = 1 : j
            z(m, 1) = Wnn1(:,m)'* images(:,i) + bnn1(1, m);
            z(m, 1) = sigmoid(z(m,1)); % sigmoid
    end
    
    a = bsxfun(@plus, Wnn2' * z, bnn2');
    y = zeros(k, 1);
    exp_ak = exp(a);
    sigma_ak = sum(exp(a));
    for m = 1 : k
        y(m, 1) = exp_ak(m, 1) / sigma_ak;
    end
    
    % back propagation
    del_k = y - T(:, i);
    del_j = sigmoidGradient(z) .* (Wnn2 * del_k);
    grad1 = images(:,i) * del_j';
    grad2 = z * del_k';
    
    Wnn1 = Wnn1 - etaNN * grad1;
    Wnn2 = Wnn2 - etaNN * grad2;
end

save('proj3.mat');