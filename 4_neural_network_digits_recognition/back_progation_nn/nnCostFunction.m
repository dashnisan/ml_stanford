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
% Part 1: Feedforward the neural network WITH DUMMY theta1 and theta2 and 
%					return the cost in the variable J. After implementing Part 1, you 
%					can verify that your cost function computation is correct by verifying 
%					the cost computed in ex4.m
%
% Part 2: Regularization of J.
%
% Part 3: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 3, you can check
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
% Part 4: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 3.
%

% ====================== PART 1: FWD J(THETA) ======================
% 
% Calculus of extended matrix X (add column of 1s), vectors A^(l) l=1,2 and matrix
% YK with logical output column vectors:

%display(['\n Theta1 size: ', num2str(size(Theta1))])
%display(['\n Theta2 size: ', num2str(size(Theta2))])

% Computation of h_theta(x^(i)) for all training examples i  (vector form):

X=[ones(m,1) X];
%display(['\n X size: ', num2str(size(X))])
% (II): Compute a's in layer 2:

A2=sigmoid(Theta1*X');

% (III): Add a_0 in layer 2 as a row of 1s:

A2=[ones(1,m); A2];
%display(['\n A2 size: ', num2str(size(A2))])

% (IV): Compute h_(THETA)(x^(i)) for all i / a's in layer 3 (output/prediction):

A3=sigmoid(Theta2*A2); 
%display(['\n A3 size: ', num2str(size(A3))])
%display(['\n mean(A3): ', num2str(mean(A3,1))])
%-------------------------------------------------------------------

% LOOP over training examples to calculate matrix YK:

for k=1:num_labels

	% KV is a vector with the classes/digits 1 to 10:	
  KV(k,1)=k; % When marking, teacher changes number of classes, so KV must be created in loop, not 			manually.

	% YK is a matrix whose columns are the logical vectors Y_k:
  YK(:,k) = y==KV(k); % Generation of logical arrays y_k=1,... y_k=num_labels. 
  %display(['YK size=',num2str(size(YK))])


end
%display(['\n YK size: ', num2str(size(YK))])
%display(['\n mean(YK): ', num2str(mean(YK))])
%-------------------------------------------------------------------

%======= J(THETA) UNREGULARIZED Vectorizing on i, summing on k: ===============

LA3=log(A3);
L1sA3=log(1-A3);
YK1s=1-YK;
sum1=0;
sum2=0;
for k=1:num_labels
	sum1=sum1+(LA3(k,:)*YK(:,k));
	sum2=sum2+(L1sA3(k,:)*YK1s(:,k));
end

J=-1/m*(sum1+sum2); % No Regularized J(THETA)

% ====================== PART 2: FWD J(THETA) REGULARIZED ======================

sum3=sum(sum(Theta1(:,[2:size(Theta1,2)]).^2));
sum4=sum(sum(Theta2(:,[2:size(Theta2,2)]).^2));

J=J+lambda*(sum3+sum4)/(2*m); % Regularized J(THETA)

% ====================== PART 3: BACK: GRADIENTS dJ(THETA)/d(THETA) UNREGULARIZED ===================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

delta3=A3-YK';
%display(['\n delta3 size: ', num2str(size(delta3))])

%display(['size theta2T*delta3',num2str(size(Theta2'*delta3))])
%display(['size theta2T*delta3:',num2str(size((Theta2(:,[2:end]))'*delta3))])
%display(['size sigmoidGradient(Theta1*X): ',num2str(size(sigmoidGradient(Theta1*X')))])


delta2=((Theta2(:,[2:end]))'*delta3).*sigmoidGradient(Theta1*X');
%display(['\n delta2 size: ', num2str(size(delta2))])

% Loop over training examples:

Bdelta1=0;
Bdelta2=0;

for i=1:m
	% Calculation of

	%Bdelta1=Bdelta1+delta2(:,i)*(X(i,[2:end])); % without bias is wrong!
	Bdelta1=Bdelta1+delta2(:,i)*(X(i,:));   

	%Bdelta2=Bdelta2+delta3(:,i)*(A2([2:end],i))'; % without bias is wrong!
	Bdelta2=Bdelta2+delta3(:,i)*(A2(:,i))';  
end

Theta1_grad=(1/m)*Bdelta1;
mean(mean(Theta1_grad))
%display(['\n Bdelta1 size: ', num2str(size(Bdelta1))])

Theta2_grad=(1/m)*Bdelta2;
mean(mean(Theta2_grad))
%display(['\n Bdelta2 size: ', num2str(size(Bdelta2))])

grad=[Theta1_grad(:); Theta2_grad(:)];

% ====================== PART 3: BACK: GRADIENTS dJ(THETA)/d(THETA) REGULARIZED ===================

% Gradient Regularizing Term (GRT):

GRT1=cat(2, zeros(size(Theta1,1),1), Theta1(:,[2:end]));
GRT2=cat(2, zeros(size(Theta2,1),1), Theta2(:,[2:end]));

Theta1_grad=Theta1_grad+lambda*GRT1/m;
Theta2_grad=Theta2_grad+lambda*GRT2/m;

grad=[Theta1_grad(:); Theta2_grad(:)];

% Single instruction for all l, independent on number of hidden layers:
%TH=0;
%for l=1:hidden_layer_size
	
%	TH=Theta.... TO BE FINISHED!!!!!!!!!!!!
%	GRT{i,1}=lambda*cat(2, zeros(size(TH,1),1), TH(:,[2:end]))

%end
