%% Project 3 - Classification
%% Nandakishore Krishna
%% Person number : 50169797

clear; close all; clc;

UBitName = 'Nanda Kishore Krishna';
personNumber = '50169797';

format long g

% training set
train_x = loadMNISTImages('../data/train-images.idx3-ubyte');
train_x = reshape(train_x,28,28,60000);

labels = loadMNISTLabels('../data/train-labels.idx1-ubyte');

% 10 digits
k = 10;

% target matrix, label 0 is mapped to 1, label 1 to 2 and so on
train_y = zeros(k, length(labels));
for i = 1 : k
    train_y(i, :) = (labels == (i-1));
end

% validation set
test_x = loadMNISTImages('../data/t10k-images.idx3-ubyte');
test_x = reshape(test_x,28,28,10000);
valLabels = loadMNISTLabels('../data/t10k-labels.idx1-ubyte');

% target matrix, label 0 is mapped to 1, label 1 to 2 and so on
test_y = zeros(k, length(valLabels));
for i = 1 : k
    test_y(i, :) = (valLabels == (i-1));
end

%% ex1 Train a 6c-2s-12c-2s Convolutional neural network 
%will run 1 epoch in about 200 second and get around 11% error. 
%With 100 epochs you'll get around 1.2% error
rand('state',0)
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};
cnn = cnnsetup(cnn, train_x, train_y);

opts.alpha = 1;
opts.batchsize = 50;
opts.numepochs = 10;

cnn = cnntrain(cnn, train_x, train_y, opts);

[er, bad] = cnntest(cnn, test_x, test_y);

%plot mean squared error
figure; plot(cnn.rL);
xlabel('iterations', 'Color','r');
ylabel('classification error', 'Color', 'r');

assert(er<0.12, 'Too big error');

save('proj3_cnn.mat');