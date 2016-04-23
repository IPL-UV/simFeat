% Noise estimation as the distance to the average of the nearest neighbors
% Inputs:
%       -X    : Original data. Matrix, M(samples) x F(features).
%       -Mn   : # of nearest neighbors
%
% Outputs:
%       -N   : Noise estimation. Matrix, M(samples) x F(features).

function N = noise(X,Nn)

A = pdist(X);
D = squareform(A);
[~,ixd] = sort(D);

N = zeros(size(D,1), size(X,2));

for t = 1:size(D,1)
    V = X(ixd(2:Nn+1,t),:);
    X1 = mean(V);
    N(t,:) = X(t,:) - X1;
end
