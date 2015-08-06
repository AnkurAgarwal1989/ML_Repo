function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

p_error = 10e9;
exponents = -2: 1;
%values_to_try = repmat(10, 1, length(exponents)); 
%values_to_try = values_to_try .^ exponents;
values_to_try = [0.01 0.03 0.1 0.3 1 3 10 30];
for C_i= values_to_try
    for sigma_j= values_to_try
        % find a model using the training data set given...
        model= svmTrain(X, y, C_i, @(x1, x2) gaussianKernel(x1, x2, sigma_j));
        
        %see how that model performs on the cross-val data
        pred = svmPredict(model, Xval);
        
        prediction_error = mean(double(pred ~= yval));
        
        if (prediction_error < p_error)
            p_error = prediction_error;
            C = C_i;
            sigma = sigma_j;
        end
    end
end


% =========================================================================

end
