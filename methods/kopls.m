% Compute the principal components of KOPLS method.
%
% Inputs:
%       - X     : Original data. Matrix, M(samples) x N(features)
%       - Yb    : Labeled of X. Matrix, M(samples) x C(classes)
%       - Nfmax : # features extracted
%
% Outputs:
%       - U     : Struct:
%                   - basis  : principal componets.Matrix, M(samples) x R(rank(K*Y))
%                   - train  : training original data
%                   - method : feature extraction method
%                   - kernel : Kernel kind
%                   - Ktrain : Kernel train

function U = kopls(X, Y, Nfeat, estimateSigmaMethod)
% KOPLS: K * Ky * K * U_kopls = s * K * K * U_kopls

% Rough estimation of the sigma parameter:
if ~exist('estimateSigmaMethod', 'var'),
    estimateSigmaMethod = 'mean';
end
sigmax = estimateSigma(X, [], estimateSigmaMethod);

% Build kernel train
K = kernel('rbf', X, X, sigmax);

[U_kopls d] = gen_eig(K * (Y * Y') * K, K' * K, Nfeat);

U.lambda = d;
U.basis = U_kopls;
U.method = 'KOPLS';
U.train = X;
U.Ktrain = K;
U.kernel = 'rbf';
U.sigma = sigmax;
