function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(1500, 1);

for iter = 1:1500

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
J =0;
for i = 1:m
  J = J + ((theta(1,1)*X(i,1) + theta(2,1)* X(i,2)) - y(i)) * X(i,1)
end
J = J/m;
theta(1,1) = theta(1,1) - alpha *  J;
 K = 0;
for i = 1:m
  K = K + ((theta(1,1)*X(i,1) + theta(2,1)* X(i,2)) - y(i)) * X(i,2)
 end
  K = K/m;
  
theta(2,1) = theta(2,1) - alpha *  K;




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
