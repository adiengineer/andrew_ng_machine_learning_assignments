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
logRegHyp=sigmoid(X*theta); %hypothesis matrix for logical regression.
%need to take log on the preceding matrix.

%add the regularizing terms
addtoJ=theta;
addtoJ=addtoJ.^2;
%t1=sum(addtoJ)-addtoJ(1);
J=(-1/m)*( y'*log(logRegHyp) + (ones(length(y),1)-y)'*log(ones(length(y),1)-logRegHyp)) + (lambda/(2*m))*(sum(addtoJ)-addtoJ(1)); %computing the cost for the given theta params, minus removed

grad=(X'*( logRegHyp-y ))/m + ((lambda/m) *theta); %gradient matrix is obtained using vectorization can be seen if we expand out the summation.
grad(1)=grad(1)-(lambda/m*theta(1));




% =============================================================

end
