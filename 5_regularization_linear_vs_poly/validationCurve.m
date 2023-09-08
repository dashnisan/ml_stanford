function [lambda_vec, error_train, error_val, JCVmin, thetaJCVmin, lambda_optim] = ...
    validationCurve(X, y, Xval, yval)

% BY DIEGO:
% I added following outputs for easier calculation of J for test-set with optimal lambda value:
% JCVmin: minimum CV-error among outputs with different lambda values
% thetaJCVmin: parameters for this minimum value. 
% lambda_optim: lambda for this minimum value.

%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               lambda = lambda_vec(i)
%
% Note: You can loop over lambda_vec with the following:
%
%       for i = 1:length(lambda_vec)
%           lambda = lambda_vec(i);
%           % Compute train / val errors when training linear 
%           % regression with regularization parameter lambda
%           % You should store the result in error_train(i)
%           % and error_val(i)
%           ....
%           
%       end
%
%

for i = 1:length(lambda_vec)
	% Train model with lambda non zero:
	theta{i,1}=trainLinearReg(X, y, lambda_vec(i));

	% Get J_train for this subset sing calculated theta{i,1} and lambda=0:
	error_train(i)=linearRegCostFunction(X, y, theta{i,1}, 0);

	% Get J_cv for the whole cv-set using calculated theta{i,1} and lambda=0:
	error_val(i)=linearRegCostFunction(Xval, yval, theta{i,1}, 0);	

end

% Passing necessary values for calculation of Jtest with optimum lambda value:

[JCVmin imin]=min(error_val); % Minimun J_cv and its index position

lambda_optim=lambda_vec(imin); % Lambda for min J_cv

thetaJCVmin = theta{imin,1}; % theta for

display(['lambda_optim=',num2str(lambda_optim)])






% =========================================================================

end
