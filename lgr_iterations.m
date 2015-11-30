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

% LR biases 1 x K
blr = 0.1 * ones(1, k);

% learning rate
eta = 10^-3;
epochs = [1, 2, 3, 4, 5, 6, 7, 8, 20, 50];

% error
lgr_error = zeros(length(epochs),1);

for e = 1 : length(epochs)
    
    % Logistic regression weights D x K
    Wlr = [blr; zeros(d, k)];
    
    % gradient descent
    for i = 0 : length(images) * epochs(1,e) - 1
        j = mod(i, length(images)) + 1;
        a = Wlr' * images(:, j);
        y = zeros(k, 1);
        exp_a = exp(a);
        sigma_a = sum(exp(a));
        for m = 1 : k
            y(m, 1) = exp_a(m, 1) / sigma_a;
        end
        Wlr = Wlr - eta * ( images(:, j) * (y - T(:, j))' );
    end
    
    % validate the weights
    predictLGR = Wlr' * valImages;
    [~, c] = max(predictLGR, [], 1);
    c = (c - 1)';
    
    lgr_error(e,1) = sum(c ~= valLabels) / size(valLabels, 1);
    fprintf('epoch %d, error %f\n', e, lgr_error(e,1));
end

figure(10)
plot(epochs, lgr_error);
xlabel('epochs', 'Color','r');
ylabel('classification error rate', 'Color', 'r');

save('lgr_iterations.mat');