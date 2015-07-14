function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
len_theta = length(theta);
J_history = zeros(num_iters, 1);
theta_temp = zeros(len_theta, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    difference = X*theta - y;
    for theta_iter = 1: len_theta
        theta_temp(theta_iter) = theta(theta_iter) - alpha * (sum(difference.* X(:, theta_iter)))/m;
    end 
    
    for theta_iter = 1: len_theta
        theta(theta_iter) = theta_temp(theta_iter);
    end
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
