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


X = [ones(m, 1) X]

% Forward propagation
a_1 = X    % 5000 * 401
z_3 = Theta1 * a_1'     % (25 * 401) * (401 * 5000) = 25 * 5000
a_2 = [ones(m,1) sigmoid(z_3)']       %26 * 5000
z_3 = Theta2 * a_2'     % (10 * 26) * (26 * 5000) = 10 * 5000
h_theta = sigmoid(z_3)      % 10 * 5000

y_new = zeros(num_labels, m)        % 10 * 5000

for i=1:m
  y_new(y(i),i)=1;
end

J = (1/m) * sum(sum((-y_new) .* log(h_theta) - (1 - y_new) .* log(1 - h_theta)))

% For Regulized Cost Function
    t_1 = Theta1(:, 2: size(Theta1,2))
    t_2 = Theta2(:, 2: size(Theta2,2))
    
    reg_val = (lambda /(2*m))* (sum(sum(t_1 .^2)) + sum(sum(t_2 .^ 2)))
    
    J = J + reg_val


% For Backward Propagation    
for i=1:m
    % 1
    a_1 = X(i, :)   %  1 * 401
    z_2= Theta1 * a_1'     % (25 * 401) * (401 * 1) = 25 * 1
    a_2 = sigmoid(z_2) % 26 * 1
   % size(a_2)
    a_2 = [1; a_2]
    z_3= Theta2 * a_2      % (10 * 26) * (26 * 1) = 10 * 1
    a_3 = sigmoid(z_3)      % 10 * 1
    
    % 2
    d_3 = a_3 - y_new(:, i)     % 10 *1
   % size(z_2)
    z_2 = [1; z_2]  % 26 * 1
    d_2 = (Theta2' * d_3) .* sigmoidGradient(z_2)    % (26 * 10) * (10 *1) = (26 * 1) + (26 *1)= 26 * 1
    
    % 3
    d_2 = d_2(2 : length(d_2))        % 25 *1
    Theta2_grad = Theta2_grad + d_3 * a_2'    %(10 *1) * (1 * 26) = 10 *26 
    Theta1_grad = Theta1_grad + d_2 * a_1    % (25 * 1) *( 1 * 401) = 25 * 401
      
    
 end

% Step 5
Theta2_grad = (1/m) * Theta2_grad; % (10*26)
Theta1_grad = (1/m) * Theta1_grad; % (25*401)


% Regularizaton
%Theta1_grad(:, 1) = Theat1_grad(:, 1)/m    % j=0
Theta1_grad(:, 2:end) = Theta1_grad(:, 2 : end) + lambda * Theta1(:, 2: end)/m     % j>=1 
%Theta2_grad(:, 1) = Theat2_grad(:, 1)/m    % j=0
Theta2_grad (:, 2:end)= Theta2_grad(:, 2:end) + lambda* Theta2(:, 2: end)/m     % j>=1

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
