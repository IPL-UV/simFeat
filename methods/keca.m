% Compute the principal components of KECA method.
%
% Inputs:
%       - X     : Original data. Matrix, M(samples) x N(features)
%       - Xtest : Original test data. Matrix, N(samples) x F(features)
%       - Nfmax : # features extracted
%
% Outputs:
%       - U     : Struct:
%                   - basis  : principal components.Matrix, M(samples) x R(rank(K))
%                   - train  : training original data
%                   - method : feature extraction method
%                   - kernel : Kernel type
%                   - Ktrain : Kernel train

function U = keca(X, Nfeat, estimateSigmaMethod)
% KECA: (Information Theory): Selects the principal directions that
% maximize the entropy. That is, the directions which have more
% information.

% Rough estimation of the sigma parameter:
if ~exist('estimateSigmaMethod', 'var'),
    estimateSigmaMethod = 'mean';
end
sigmax = estimateSigma(X, [], estimateSigmaMethod);

% Build kernel train
K = (1/sqrt(2*sigmax*pi)) * kernel('rbf', X, X, sigmax);

[U,s] = eigs(K,Nfeat);
m = zeros(1,size(U,2));
for t = 1:size(U,2)
    m(t) = (sqrt(diag(s(t,t))) .* U(:,t)' * ones(size(U,1),1))^2;
end

[~,ind] = sort(m,'descend');
U_keca = U(:,ind);
s = s(ind,ind);

U.lambda = s(ind,ind);
U.basis = U_keca;
U.method = 'KECA';
U.train = X;
U.Ktrain = K;
U.kernel = 'rbf';
U.sigma = sigmax;
