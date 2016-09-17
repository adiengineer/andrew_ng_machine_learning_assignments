function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
hypval=X*theta; %contains the values which the hypothesis takes at each training sample point.

J=(sum( ( hypval-y ).^2 ))/( 2*m ); %computing the squared error using vectorization.


% =========================================================================

end
