function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

%sigmoidVal = transpose(theta)*X(i);
for i = 1:m
sigmoidVal = X(i,:)*theta;
J = J + [y(i)* log(sigmoid(sigmoidVal)) + (1 - y(i)) * log(1 - sigmoid(sigmoidVal))]
end
J = -J/m;

count = 0
for t = 1:size(theta)
  temp = 0;
  for i = 1:m
    sigmoidVal = X(i,:)*theta;
    %fprintf('sigmoid val:%f',sigmoid(sigmoidVal));
    count = count+1;
    %fprintf('Value of sig func : %f\n',(sigmoid(sigmoidVal) - y(i))*X(i,t));
    temp += ((sigmoid(sigmoidVal) - y(i))*X(i,t));
  end
  %temp;
  %
  grad(t) = temp/m;
  %temp=0;
  %fprintf('grad(t) : %f',grad(t)); 
end
  





% =============================================================

end
