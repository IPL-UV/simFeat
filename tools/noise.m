% Noise estimation as the distance to the average of the nearest neighbors
% Inputs:
%       -X    : Original data. Matrix, M(samples)xF(features).
%       -Mn   : # of nearest neighbors
%
% Outputs:
%       -N   : Noise estimation. Matrix, M(samples)xF(features).

function N=noise(X,Nn)

A=pdist(X);
D=squareform(A);
[v,ixd]=sort(D);
for t=1:size(D,1)
    V=X(ixd(2:Nn+1,t),:);
    X1 = mean(V);
    N(t,:)=X(t,:)-X1;
end