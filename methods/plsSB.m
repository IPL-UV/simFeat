% Compute the principal components of PLS method.
% Inputs:
%       -X    : Original data. Matrix, M(samples)xN(features).
%       -Yb   : Labeled of X. Matrix, M(samples)xC(classes).
%       -Nfmax: # features extracted.
%
% Outputs:
%       -U    : Struct:
%                       -basis  : principal componets. Matrix, M(samples)xR(rank(Cxy)).
%                       -train  : training original data
%                       -method : feature extraction method

function U=plsSB(X,Y,Nfmax)
% PLS: Cxy*U_pls=s*U_pls

%-----------------
%  PLS
%-----------------
Yb = binarize(Y); % Encode the labels with a 1-of-C scheme

Cxy=X'*Yb;

[A,S,V] = svds(Cxy,min(Nfmax,rank(Cxy)));

U.lambda=S;
U.basis=A;
U.method='PLS-SB';
U.train=X;
