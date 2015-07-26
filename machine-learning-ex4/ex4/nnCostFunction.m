function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%=======================================================================
%Part 1
% Recoding output y to label matrix (Y)
y_eye = eye(num_labels);
Y = y_eye(y, :);

%Feed forward
% add bias unit
%X = [ones(m,1) X];
z2 = [ones(m,1) X] * Theta1'; % we need to add bias units only for feed-forward
a2 = sigmoid(z2);

%add bias unit
%a2 = [ones(size(a2,1),1) a2];
z3 = [ones(size(a2,1),1) a2] * Theta2'; % we need to add bias units only for feed-forward
a3 = sigmoid(z3); 

%Cost calculation
% J = 1/m ((-Y(ln(prediction)) - (1-Y)(ln(1-prediction)))
term1 = -Y .* (log(a3));
term2 = -(1-Y) .* (log(1 - a3));
sum_over_labels = sum((term1 + term2), 2); % this will return mx1 vector
J = mean(sum_over_labels); % we need to do 1/m sum (which will be mean)

%=======================================================================
%Part 3
% Regularization
reg_Theta1 = sum(sum(Theta1(:, 2:end) .^ 2)); % summing up of squares of all theta values...leave out bias weights which will be in column 1
reg_Theta2 = sum(sum(Theta2(:, 2:end) .^ 2));
reg = (lambda) * (reg_Theta1 + reg_Theta2)/(2*m);

J = J + reg;
% -------------------------------------------------------------

%=======================================================================
%Part 2
% Back Propagation
del3 = a3 - Y; %PD of Cost wrt Theta2

% Code logic...since we multiplied inputs by (Theta') to calculate activations...
%we will use Theta to back prop deltas.
del2 = (del3 * Theta2(:, 2:end)) .* sigmoidGradient(z2); %PD of Cost wrt Theta3
%leaving first column from Theta2 allows us to not calculate
%since first values are just biases and are always 1...
% =========================================================================

% Unroll gradients
Theta1_grad = del2' * [ones(m,1) X]; %add the bias column to inputs
Theta2_grad = del3' * [ones(size(a2,1),1) a2] ; %add the bias column to inputs
grad = 1/m * [Theta1_grad(:) ; Theta2_grad(:)];

end
