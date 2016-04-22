% Compute the principal components of OPLS method.
% Inputs:
%       -X    : Original data. Matrix, M(samples)xN(features).
%       -Yb   : Labeled of X. Matrix, M(samples)xC(classes).
%       -Nfeat: # features extracted.
%
% Outputs:
%       -U    : Struct:
%                       -basis  : principal componets. Matrix, M(samples)xR(rank(Cxy)).
%                       -train  : training original data
%                       -method : feature extraction method

function U = opls(X,Y,Nfeat)
% OPLS: Cxy * Cxy' * U_opls = s * Cxx * U_opls
Yb = binarize(Y); % Encode the labels with a 1-of-C scheme

Cxx = X' * X;
Cxy = X' * Yb;

[U_opls d] = gen_eig(Cxy * Cxy', Cxx, Nfeat);

U.lambda = d;
U.basis = U_opls;
U.method = 'OPLS';
U.train = X;
