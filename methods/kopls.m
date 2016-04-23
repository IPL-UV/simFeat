% Compute the principal components of KOPLS method.
%
% Inputs:
%       -X    : Original data. Matrix, M(samples)xN(features).
%       -Yb   : Labeled of X. Matrix, M(samples)xC(classes).
%       -Nfmax: # features extracted.
%
% Outputs:
%       -U    : Struct:
%                       -basis  : principal componets.Matrix, M(samples)xR(rank(K*Y)).
%                       -train  : training original data
%                       -method : feature extraction method
%                       -kernel : Kernel kind.
%                       -Ktrain : Kernel train.

function U = kopls(X, Y, Nfeat, estimateSigmaMethod)
% KOPLS: K * Ky * K * U_kopls = s * K * K * U_kopls

Yb = binarize(Y); % Encode the labels with a 1-of-C scheme

% Rough estimation of the sigma parameter:
% sigmax = estimateSigma(X,X);
if ~exist('estimateSigmaMethod', 'var'),
    estimateSigmaMethod = 'mean';
end
sigmax = estimateSigma(X, [], estimateSigmaMethod);

% Build kernel train
K = kernel('rbf', X, X, sigmax);

[U_kopls d] = gen_eig(K * (Yb * Yb') * K, K' * K, Nfeat);

U.lambda = d;
U.basis = U_kopls;
U.method = 'KOPLS';
U.train = X;
U.Ktrain = K;
U.kernel = 'rbf';
U.sigma = sigmax;
