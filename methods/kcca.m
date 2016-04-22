% Compute the principal components of KCCA method.
%
% Inputs:
%       -X    : Original data. Matrix, M(samples)xN(features).
%       -Y    : Labeled of X. Vector, M(samples)x1.
%       -Nfmax: # features extracted.
%
% Outputs:
%       -U    : Struct:
%                       -basis  : principal componets.Matrix, M(samples)xR(rank(K*Y)).
%                       -train  : training original data
%                       -method : feature extraction method
%                       -kernel : Kernel kind.
%                       -Ktrain : Kernel train.

function U = kcca(X,Y,Nfeat)

Yb = binarize(Y); % Encode the labels with a 1-of-C scheme

% Rough estimation of the sigma parameter:
sigmax = estimateSigma(X,X);

% Build kernel train
K  = kernel('rbf',X,X,sigmax);
Kc = kernelcentering(K);

% Centered labels
[n dy] = size(Yb);
Y1 = Yb - repmat(mean(Yb),length(Y),1);

% solving the generalized eigenproblem is a nightmare
% dx = size(Kc,2);
% dy = size(Y1,2);
% KK = Kc*Kc;
% KY = Kc*Y;
% Cyy = Y1'*Y1;
% A = [zeros(dx,dx)     KY;     
%      KY'     zeros(dy,dy)];
% B = [KK     zeros(dx,dy);     
%      zeros(dy,dx)     Cyy];
% [U_kcca d]= gen_eig(A,B,Nfeat);
% [U_kcca d]= eig(A,B);
% it is much simpler is to conver it into a single eigenproblem

Cxx = Kc * Kc + 1e-6* eye(n);
Cyy = Y1' * Y1 + 1e-6 * eye(dy);
% [U_kcca, lambda] = gen_eig(Kc * Y1 * inv(Cyy) * Y1' * Kc, Cxx, Nfeat); % Basis in X
[U_kcca, lambda] = gen_eig(Kc * Y1 * (Cyy \ Y1') * Kc, Cxx, Nfeat); % Basis in X
D         = sqrt(lambda);      % Canonical correlations
% V = gen_eig(Y1' * Kc * inv(Cxx) * Kc * Y1, Cyy, Nfeat); % Basis in Y
V = gen_eig(Y1' * Kc * (Cxx \ Kc) * Y1, Cyy, Nfeat); % Basis in Y

U.lambda  = D;
U.basis   = U_kcca;
U.V       = V;
U.method  = 'KCCA';
U.train   = X;
U.Ktrain  = K;
U.kernel  = 'rbf';
U.sigma   = sigmax;
