function g = sigmoidGradient(z)
    temp = sigmoid(z);
    g = temp .* (1.0 - temp);
end