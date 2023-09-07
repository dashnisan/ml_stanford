function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

 
%display(num2str(theta))
theta

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    A=theta'*X'; % 1xm 
    %display(num2str(size(A)))
    B=A'-y; % 1xm
    %display(num2str(size(B)))
    %display(num2str(size(X)))
    C=B'*X; %1xn 
    %display(num2str(size(C)))
    prefix=-alpha/m;
    dth=prefix*C;
   % display(num2str(size(theta)))
    %display(num2str(size(dth)))

    theta=theta+dth';
    %display(num2str(theta))
   %display(num2str(iter))

    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    %display(num2str(J_history))

end
%plot(J_history)
%end
