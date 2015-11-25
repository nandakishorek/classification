function [J, grad] = errorFunction(W, d, j, k, bnn1, bnn2, X, y)
    
    Wnn1 = reshape(W(1:j * d), ...
                 d, j);

    Wnn2 = reshape(W((1 + (d * j)):end), ...
                 j, k);

    % feed forward propagation
    z = bsxfun(@plus, Wnn1' * X , bnn1');
    z = sigmoid(z); % sigmoid
       
    a = bsxfun(@plus, Wnn2' * z, bnn2');
    yk = zeros(k, length(y));
    exp_ak = exp(a);
    sigma_ak = sum(exp(a));
    for l = 1 : length(y) 
        for m = 1 : k
            yk(m, l) = exp_ak(m, l) / sigma_ak(1, l);
        end
    end
    
    J = sum(-1 * sum(y .* log(yk))) / length(y);
    
    % back propagation
    del_k = yk - y;
    del_j = sigmoidGradient(z) .* (Wnn2 * del_k);
    grad1 = X * del_j';
    grad2 = z * del_k';
    
    grad = [grad1(:); grad2(:)];
    fprintf('returning from error')
end
