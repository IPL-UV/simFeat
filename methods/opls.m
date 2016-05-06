% Compute the principal components of OPLS method.
% Inputs:
%       - X     : Original data. Matrix, M(samples) x N(features)
%       - Yb    : Labeled of X. Matrix, M(samples) x C(classes)
%       - Nfeat : # features extracted
%
% Outputs:
%       - U     : Struct:
%                   - basis  : principal componets. Matrix, M(samples) x R(rank(Cxy))
%                   - train  : training original data
%                   - method : feature extraction method

function U = opls(X,Y,Nfeat)
% OPLS: Cxy * Cxy' * U_opls = s * Cxx * U_opls

Cxx = X' * X;
Cxy = X' * Y;

[U_opls d] = gen_eig(Cxy * Cxy', Cxx, Nfeat);

U.lambda = d;
U.basis = U_opls;
U.method = 'OPLS';
U.train = X;
