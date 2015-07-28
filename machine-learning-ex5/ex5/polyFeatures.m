function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%

% we dont need to use 0 power..as we add the 1 term to X in the ex5
% You need to return the following variables correctly.

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% 
X = repmat(X, 1, p);
p = repmat((1:p), size(X,1), 1);
X_poly = X .^ p;




% =========================================================================

end
