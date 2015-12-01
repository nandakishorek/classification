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

% % normalize
% [images, mu, sigma] = zscore(images');
% images = images';

% append extra 1's row
images = [ones(1, size(images,2)); images];

% target matrix, label 0 is mapped to 1, label 1 to 2 and so on
T = zeros(k, length(labels));
for i = 1 : k
    T(i, :) = (labels == (i-1));
end

% validation set
valImages = loadMNISTImages('../data/t10k-images.idx3-ubyte');
% valImages = normalize(valImages', mu, sigma);
% valImages = valImages';

% append extra 1's row
valImages = [ones(1,size(valImages,2)); valImages];

valLabels = loadMNISTLabels('../data/t10k-labels.idx1-ubyte');

% target matrix, label 0 is mapped to 1, label 1 to 2 and so on
valT = zeros(k, length(valLabels));
for i = 1 : k
    valT(i, :) = (valLabels == (i-1));
end

d = 784;

% remove the extra 1's rows
images = images(2:end,:);
valImages = valImages(2:end,:);

% Neural net
% number of hidden units
j = [5; 10; 15; 30; 35; 40; 45; 55];
nn_error = zeros(length(j), 1);

for e = 1 : length(j)
    % bias for first layer 1 x j
    bnn1 = 0.5 * ones(1, j(e, 1));
    
    % bias for second layer 1 x k
    bnn2 = 0.5 * ones(1, k);
    
    % weights for first layer d x j
    rng default
    epsilon_init = sqrt(6) / sqrt(d + j(e, 1));
    Wnn1 = (rand(j(e, 1), d) * 2 * epsilon_init - epsilon_init)';
    
    % weights for first layer j x k
    epsilon_init = sqrt(6) / sqrt(j(e, 1) + k);
    Wnn2 = (rand(k, j(e, 1)) * 2 * epsilon_init - epsilon_init)';
    
    % NN gradient descent
    etaNN = 0.01;
    
    for i = 1 : length(images)
        % feed forward propagation
        z = zeros(j(e, 1), 1);
        
        for m = 1 : j(e, 1)
            z(m, 1) = Wnn1(:,m)'* images(:,i) + bnn1(1, m);
            z(m, 1) = relu(z(m,1)); % relu
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
        del_j = (z > 0) .* (Wnn2 * del_k); % z > 0 is ReLu gradient
        grad1 = images(:,i) * del_j';
        grad2 = z * del_k';
        
        Wnn1 = Wnn1 - (etaNN * grad1);
        Wnn2 = Wnn2 - (etaNN * grad2);
    end
    
    % validate the weights
    fprintf('validating weights for NN\n');
    
    predictNN = bsxfun(@plus, Wnn1' * valImages, bnn1');
    predictNN = relu(predictNN);
    predictNN = bsxfun(@plus, Wnn2' * predictNN, bnn2');
    [~, c2] = max(predictNN, [], 1);
    c2 = (c2 - 1)';
    
    nn_error(e,1) = sum(c2 ~= valLabels) / size(valLabels, 1);
    fprintf('j = %d, error = %f\n', j(e,1), nn_error(e,1));
end

figure(201);
plot(j, nn_error);
xlabel('M', 'Color', 'r');
ylabel('classification error', 'Color', 'r');

save('nn_hidden_units.mat');