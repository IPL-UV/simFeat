% Compute the principal components of PCA method.
% Inputs:
%       - Xp    : Original data. Matrix, M(samples) x N(features)
%
% Outputs:
%       - U     : Struct:
%                   - basis  : principal componets. Matrix, M(samples) x R(rank(X))
%                   - train  : training original data
%                   - method : feature extraction method

function U = pca(Xp,np)
% PCA: Cxx * U_pca = s * U_pca

N = size(Xp,2);
Cxx  = (Xp' * Xp) / (N - 1);

% [A,D,V] = eigs(Cxx,np);
[A,D] = eigs(Cxx,np);

U.lambda = D;
U.basis = A;
U.method = 'PCA';
U.train = Xp;
