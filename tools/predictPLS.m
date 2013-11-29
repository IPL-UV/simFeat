% Compute the principal components of PLS method.
% Inputs:
%       -X    : Original data. Matrix, M(samples)xN(features).
%       -Yb   : Labeled of X. Matrix, M(samples)xC(classes).
%       -Nfmax: # features extracted.
%
% Outputs:
%       -U    : Struct:
%                       -basis  : principal componets. Matrix, M(samples)xN(features)).
%                       -train  : training original data
%                       -method : feature extraction method

function [U Ypred]=predictPLS(X,Xtest,Y,Nfmax)
% PLS: Cxy*U_pls=s*U_pls

%-----------------
%  PLS
%-----------------
Yb = binarize(Y); % Encode the labels with a 1-of-C scheme

[W,A,test] = pls(X,Xtest,Yb,Nfmax);

[val,Ypred]    = max(test,[],2);

U.basis=A;
U.method='PLS';
U.train=X;
