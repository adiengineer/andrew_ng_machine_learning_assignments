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

logRegHyp=sigmoid(X*theta); %hypothesis matrix for logical regression.
%need to take log on the preceding matrix.

J=(-1/m)*( y'*log(logRegHyp) + (ones(length(y),1)-y)'*log(ones(length(y),1)-logRegHyp)); %computing the cost for the given theta params, minus removed

grad=(X'*( logRegHyp-y ))/m; %gradient matrix is obtained using vectorization can be seen if we expand out the summation.




% =============================================================

end
