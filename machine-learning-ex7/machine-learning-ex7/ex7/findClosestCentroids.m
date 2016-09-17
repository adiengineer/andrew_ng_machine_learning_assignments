function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%       idx must contain indices of the centroids as the results, not the 
%       actual coordinates of the centroids 

% Set K
K = size(centroids, 1); %each cluster has a centroid assc with it
M = length( X(:,1) ); % number of training samples

%seecentroids=centroids
% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% create distance matrtix of dimensions M X K
 distance = zeros(M,K); % for each training sample distance from K centroids

 % loop over each centroid compute distance from each training sample
 for i=1:K
     
     % for vectorization generate stack of centroids so that dist can
     % be computed for all training samples at once
  temp_centroid = repmat(centroids(i,:),M,1); % CORRECTION: included : to pick all columns
  
 % sX=size(X)
 %stemp=size(temp_centroid)
  
  distance(:,i)=sum( (X-temp_centroid).^2 , 2);
 end

 % now for each row of distance matrix min the index of minimum which will
 % correspond to the closest centroid
    [Y,I] = min(distance,[],2); % along the rows 
    
    idx= I ; 
% =============================================================

end

