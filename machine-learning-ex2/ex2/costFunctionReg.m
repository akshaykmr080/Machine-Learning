function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

for i = 1:m
sigmoidVal = X(i,:)*theta;
J = J + [y(i)* log(sigmoid(sigmoidVal)) + (1 - y(i)) * log(1 - sigmoid(sigmoidVal))]
end
J = -J/m;

thetaSum = 0;
for i = 2:length(theta)
thetaSum = thetaSum + (theta(i)*theta(i));
end

thetaSum = thetaSum *(lambda/(2*m));
 J = J  + thetaSum;
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
  if(t >1)
    grad(t) = grad(t) + (theta(t) * (lambda/m));
  
  %temp=0;
  %fprintf('grad(t) : %f',grad(t)); 
end




% =============================================================

end
