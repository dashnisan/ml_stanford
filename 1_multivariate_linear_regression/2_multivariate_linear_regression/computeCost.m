function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
%display(num2str(size(X)))
%display(num2str(size(y)))
%display(num2str(size(theta)))

%return

A=theta'*X';
B=A'-y;

% You need to return the following variables correctly 
J = (2*m)^-1*(sum(B.^2));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.





% =========================================================================

end
