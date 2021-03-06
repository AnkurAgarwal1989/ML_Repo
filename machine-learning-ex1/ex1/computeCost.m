function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
%m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
% vectorization...
% X is m x (n+1); theta is (n+1) x 1; y is m x 1
hypothesis = X*theta;
%squaring
% the computeCostMulti has another way to write this....X^2 = X' * X
error = (hypothesis - y) .^ 2;
J = mean(error)/2;


% =========================================================================

end
