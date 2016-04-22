% Compute the principal components of PLS method.
% Inputs:
%       -X    : Original data. Matrix, M(samples)xN(features).
%       -Y    : Labeled of X. Vector, M(samples)x1.
%       -Nfmax: # features extracted.
%
% Outputs:
%       -U    : Struct:
%                       -basis  : principal componets. Matrix, M(samples) x R(rank(Cxy)).
%                       -train  : training original data
%                       -method : feature extraction method

function U = cca(X, Y, Nfeat)
% CCA: A * U_cca = s * B * U_cca

Yb = binarize(Y); % Encode the labels with a 1-of-C scheme

% Centered labels
Y1 = Yb - repmat(mean(Yb),length(Y),1);

dx = size(X,2);
dy = size(Y1,2);

Cxx = X' * X + 1e-8 * eye(dx);
Cxy = X' * Y1;
Cyx = Cxy';
Cyy = Y1' * Y1 + 1e-8 * eye(dy);

% solving the generalized eigenproblem is a nightmare
% A = [zeros(dx,dx)     Cxy;     
%      Cxy'     zeros(dy,dy)];
% B = [Cxx     zeros(dx,dy);     
%      zeros(dy,dx)     Cyy];
% [U_cca d] = gen_eig(A,B,Nfeat);
% [U_cca d] = eig(A,B); 
% [U_cca d] = eigs(A,B,Nfeat);
% it is much simpler is to conver it into a single eigenproblem

%[U_cca,lambda] = gen_eig(Cxy*inv(Cyy)*Cyx, Cxx, Nfeat); % Basis in X
[U_cca,lambda] = gen_eig(Cxy * (Cyy \ Cyx), Cxx, Nfeat); % Basis in X
D = sqrt(lambda);      % Canonical correlations

% [V,l] = gen_eig(Cyx*inv(Cxx)*Cxy, Cyy, Nfeat); % Basis in Y
V = gen_eig(Cyx * (Cxx \ Cxy), Cyy, Nfeat); % Basis in Y

U.lambda = D; 
U.basis  = U_cca;
U.V      = V;
U.method = 'CCA';
U.train  = X;
