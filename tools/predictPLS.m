% Compute the principal components of PLS method.
% Inputs:
%       - X     : Original data. Matrix, M(samples) x N(features)
%       - Yb    : Labeled of X. Matrix, M(samples) x C(classes)
%       - Nfmax : # features extracted
%
% Outputs:
%       - U     : Struct:
%                   - basis  : principal componets. Matrix, M(samples) x N(features))
%                   - train  : training original data
%                   - method : feature extraction method

function [U Ypred] = predictPLS(X, Xtest, Y, Nfmax)
% PLS: Cxy * U_pls = s * U_pls

[~, A, test] = pls(X, Xtest, Y, Nfmax);
[~, Ypred] = max(test, [], 2);

U.basis = A;
U.method = 'PLS';
U.train = X;
