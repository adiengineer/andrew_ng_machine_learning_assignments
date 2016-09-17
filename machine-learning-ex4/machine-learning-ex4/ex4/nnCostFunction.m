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
%   partial derivatives of the neural network. %later unroll part deriva for
%   all terms of all matrices.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1); % no of training examples
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% add column of ones to the X matrix. These are the bis terms added to each
% training sample.
X = [ones(m, 1) X]; %bias added

% now implement the feedforward prop to finally genrate hyp vecs(repre prop
%of ex belonging to a class out of K) for each training sample.

act2=sigmoid( X * (Theta1') ); %this is activation vectors for layer 2 for each training sample.

% add ones again for next step of propogation
act2 = [ones(m, 1) act2]; % bias terms added

act3=sigmoid( act2 * (Theta2') ); %output hyp vecs for each traing samples arranged along rows.

%now loop to compute the cost by picking one hyp at a time and generating
%appropriate y vec for each training sample.

tempcost=0; % remember to divide by m
for i=1:m %looping over each training example
    %generate appropriate y vec for sample and extract hyp for this
    %sample
     Y_i=zeros(num_labels,1); % use for encoding each result
     Y_i( y(i) ) = 1 ; % that index should be 1
     hyp_i= act3(i,:); %extracting the hypo for the ith sample.
    
    tempcost= tempcost + log(hyp_i)*(Y_i) + ( log(1- hyp_i)*(1 - Y_i) ); % transpose of the matrix y 
end

%J= (-1/m)*tempcost; earlier cost without the regularization term

% now add the regularization terms
% remove the bias column from both matrices and square them and add them up
% assuming that the input layer size is same as number of features.
 reg_term=(lambda/(2*m))*( sum( sum( ( Theta1( :, 2:(input_layer_size+1) ) ).^2) )+ sum( sum( ( Theta2( :, 2:(hidden_layer_size+1) ) ).^2) ) );

J= (-1/m)*tempcost + reg_term;

% part 2 back prop

%setting gradient accumulators for Theta1 and Theta2
D1=0; D2=0;

%for loop for each training sample
for t=1:m
  % use the first training example from X.
  % use code written above, rem to add bias term
  example=X(t,:); % tth example, already has bias term added.
  
 % che1=size(Theta1)
  %che2=size(example)
  z2=Theta1* (example');  %trying to add a bias term to z2
  act2=sigmoid(Theta1* (example') ); %this is activation vectors for layer 2 for each training sample.

% add ones again for next step of propogation
%act2 = [ones(m, 1) act2];
z2= [1 ; z2]; % bias terms added
 act2= [1 ; act2]; % bias terms added

 %stheta2=size(Theta2)
 %sact2=size(act2)
act3=sigmoid(  (Theta2)*(act2)  ); %output hyp vecs for each traing samples arranged along rows

% generate the logical vector to indicate to which category does the
% example result belongs to.
 Y_t=zeros(num_labels,1); % use for encoding each result
     Y_t( y(t) ) = 1 ; % that index should be 1

  del3=act3 - Y_t ;
  
  %stheta2=size((Theta2'*del3))
 %sact2=size(sigmoidGradient(z2))
  del2= (Theta2'*del3).* sigmoidGradient(z2);
  del2 = del2(2:end); % we must neglect the bias term as is added by us never predicted so no question of asso an error with it.

  
   %sdel2=size(del2)
 %sexample=size(example)
  D1=D1 + del2*(example); % this is coz our example is a row vec and not a column vec, so removed '
  D2=D2 + del3*(act2');
  % note that there can't a del1 term as we never estimate values for the
  % input that is our raw data.
  
end

%outside the for loop
%Theta1_grad=D1/m;
%Theta2_grad=D2/m; % unregularized.
copyTheta1=Theta1*(lambda/m);
copyTheta2=Theta2*(lambda/m);
copyTheta1(:,1)=0;
copyTheta2(:,1)=0;

Theta1_grad=D1/m+copyTheta1;
Theta2_grad=D2/m+copyTheta2; %after regularization 
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
