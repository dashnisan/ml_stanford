function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


A=theta'*X'; % theta and X both colum vectors
B=A'-y;

% Regularized linear regression:

J = (2*m)^-1*(sum(B.^2) + lambda*sum(theta(2:end).^2));


% =========================================================================
% Vector form for all i (training examples) and all j (variables including x_0)

grad=(1/m)*B'*X; %1x(n+1) # Unregularized gradient

% Regularization:

dJL=lambda/m.*theta(2:end); % nx1
dJL=cat(1, 0., dJL); % (n+1)x1
grad=grad	+ dJL'; % regularized gradient


grad = grad(:);

end
