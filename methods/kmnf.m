% Compute the principal components of KMNF method.
%
% Inputs:
%       - X     : Original data. Matrix, M(samples) x N(features)
%       - Nfmax : # features extracted
%
% Outputs:
%       - U     : Struct:
%                   - basis  : principal componets.Matrix, M(samples) x Nfeat(extracted features)
%                   - train  : training original data
%                   - method : feature extraction method
%                   - kernel : Kernel type
%                   - Ktrain : Kernel train

function U = kmnf(X, Nfeat, estimateSigmaMethod)
% KMNF: Kxx * Kxx * U_mnf = s * Kxn * Kxn' * U_mnf

% Rough estimation of the sigma parameter:
if ~exist('estimateSigmaMethod', 'var'),
    estimateSigmaMethod = 'mean';
end
sigmax = estimateSigma(X, [], estimateSigmaMethod);

% Build kernel train
K = kernel('rbf', X, X, sigmax);
Kc = kernelcentering(K);

% Noise estimation
N = noise(X,10);
sigmaxn = estimateSigma(X, N, estimateSigmaMethod);
Kxn = kernel('rbf', X, N, sigmaxn);
Kxnc = kernelcentering(Kxn);

[U_kmnf d] = gen_eig(Kc * Kc, Kxnc * Kxnc', Nfeat);

U.lambda = d;
U.basis = U_kmnf;
U.method = 'KMNF';
U.train = X;
U.Ktrain = K;
U.kernel = 'rbf';
U.sigma = sigmax;
