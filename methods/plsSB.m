% Compute the principal components of PLS method.
% Inputs:
%       - X     : Original data. Matrix, M(samples) x N(features)
%       - Yb    : Labeled of X. Matrix, M(samples) x C(classes)
%       - Nfmax : # features extracted
%
% Outputs:
%       - U     : Struct:
%                   - basis  : principal componets. Matrix, M(samples) x R(rank(Cxy))
%                   - train  : training original data
%                   - method : feature extraction method

function U = plsSB(X, Y, Nfmax)
% PLS: Cxy * U_pls = s * U_pls

%-----------------
%  PLS
%-----------------
Cxy = X' * Y;

% [A,S,V] = svds(Cxy,min(Nfmax,rank(Cxy)));
[A,S] = svds(Cxy, min(Nfmax, rank(Cxy)));

U.lambda = S;
U.basis = A;
U.method = 'PLS-SB';
U.train = X;
