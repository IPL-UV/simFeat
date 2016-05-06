% Compute the principal components of KPCA method.
%
% Inputs:
%       - X     : Original data. Matrix, M(samples) x N(features)
%       - Nfmax : # features extracted
%
% Outputs:
%       - U     : Struct:
%                   - basis  : principal components. Matrix, M(samples) x R(rank(K))
%                   - train  : training original data
%                   - method : feature extraction method
%                   - kernel : Kernel type
%                   - Ktrain : Kernel train

function U = kpca(X, Nfeat, estimateSigmaMethod)
% KCCA: A * U_kcca = s * B * U_kcca

% Rough estimation of the sigma parameter:
if ~exist('estimateSigmaMethod', 'var'),
    estimateSigmaMethod = 'mean';
end
sigmax = estimateSigma(X, [], estimateSigmaMethod);

% Build kernel train
K = kernel('rbf', X, X, sigmax);
Kc = kernelcentering(K);

[U_kpca,s] = eigs(Kc, Nfeat);

U.lambda = s;
U.basis = U_kpca;
U.method = 'KPCA';
U.train = X;
U.Ktrain = K;
U.kernel = 'rbf';
U.sigma = sigmax;
