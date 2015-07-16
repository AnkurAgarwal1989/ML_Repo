function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
feature_length = size(X, 2);
mu = zeros(1, feature_length);
sigma = zeros(1, feature_length);

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       
% for iter = 1: feature_length
%     mu(iter) = mean(X_norm(:,iter));
%     sigma(iter) = std(X_norm(:,iter));
%     X_norm(:,iter) = (X_norm(:,iter) - mu(iter)) / sigma(iter);
% end
%vectorization method
mu = mean(X, 1); % columnwise mean
sigma = std(X, 1); % columnwise standard deviation
%these are vectors (row vector)...make them into arrays for faster subtraction and division w/o for loops
mu = repmat(mu, size(X,1), 1); %replicated each row...in row direction
sigma = repmat(sigma, size(X,1), 1); %replicated each row...in row direction
X_norm = (X - mu) ./ sigma;





% ============================================================

end
