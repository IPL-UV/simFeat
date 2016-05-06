% Compute the principal components of KPLS method.
%
% Inputs:
%       - X     : Original data. Matrix, M(samples) x N(features)
%       - Yb    : Labeled of X. Matrix, M(samples) x C(classes)
%       - Nfmax : # features extracted
%
% Outputs:
%       - U     : Struct:
%                   - basis  : principal componets. Matrix, M(samples) x R(rank(K*Y))
%                   - train  : training original data
%                   - method : feature extraction method
%                   - kernel : Kernel type
%                   - Ktrain : Kernel train

function U = kpls(X, Y, Nfeat, estimateSigmaMethod)
% KPLS: K * Y * U_kpls = s * U_kpls

% Rough estimation of the sigma parameter:
if ~exist('estimateSigmaMethod', 'var'),
    estimateSigmaMethod = 'mean';
end
sigmax = estimateSigma(X, [], estimateSigmaMethod);

% Build kernel train
K = kernel('rbf', X, X, sigmax);
Kc = kernelcentering(K);

% [U_kpls,s,v] = svds(Kc * Y, Nfeat);
[U_kpls,s] = svds(Kc * Y, Nfeat);

U.lambda = s;
U.basis = U_kpls;
U.method = 'KPLS';
U.train = X;
U.Ktrain = K;
U.kernel = 'rbf';
U.sigma = sigmax;
