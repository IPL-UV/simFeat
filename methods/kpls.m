% Compute the principal components of KPLS method.
%
% Inputs:
%       -X    : Original data. Matrix, M(samples)xN(features).
%       -Yb   : Labeled of X. Matrix, M(samples)xC(classes).
%       -Nfmax: # features extracted.
%
% Outputs:
%       -U    : Struct:
%                       -basis  : principal componets. Matrix, M(samples)xR(rank(K*Y))
%                       -train  : training original data
%                       -method : feature extraction method
%                       -kernel : Kernel kind.
%                       -Ktrain : Kernel train.

function U = kpls(X,Y,Nfeat)
% KPLS: K * Y * U_kpls = s * U_kpls

Yb = binarize(Y); % Encode the labels with a 1-of-C scheme

% Rough estimation of the sigma parameter:
sigmax = estimateSigma(X,X);

% Build kernel train
K = kernel('rbf',X,X,sigmax);
Kc = kernelcentering(K);

% [U_kpls,s,v] = svds(Kc * Yb,Nfeat);
[U_kpls,s] = svds(Kc * Yb, Nfeat);

U.lambda = s;
U.basis = U_kpls;
U.method = 'KPLS';
U.train = X;
U.Ktrain = K;
U.kernel = 'rbf';
U.sigma = sigmax;
