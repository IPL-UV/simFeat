% Compute the principal components of PLS method.
% Inputs:
%       - X     : Original data. Matrix, M(samples) x N(features)
%       - Y     : Labeled of X. Vector, M(samples) x 1
%       - Nfmax : # features extracted
%
% Outputs:
%       - U     : Struct:
%                   - basis  : principal componets. Matrix, M(samples) x R(rank(Cxy))
%                   - train  : training original data
%                   - method : feature extraction method

function U = cca(X, Y, Nfeat)
% CCA: A * U_cca = s * B * U_cca

Cxx = X' * X + 1e-8 * eye(size(X,2));
Cxy = X' * Y;
Cyx = Cxy';
Cyy = Y' * Y + 1e-8 * eye(size(Y,2));

% solving the generalized eigenproblem is a nightmare
% A = [zeros(dx,dx)     Cxy;     
%      Cxy'     zeros(dy,dy)];
% B = [Cxx     zeros(dx,dy);     
%      zeros(dy,dx)     Cyy];
% [U_cca d] = gen_eig(A,B,Nfeat);
% [U_cca d] = eig(A,B); 
% [U_cca d] = eigs(A,B,Nfeat);
% it is much simpler is to conver it into a single eigenproblem

% [U_cca,lambda] = gen_eig(Cxy * inv(Cyy) * Cyx, Cxx, Nfeat); % Basis in X
[U_cca,lambda] = gen_eig(Cxy * (Cyy \ Cyx), Cxx, Nfeat); % Basis in X
D = sqrt(lambda);      % Canonical correlations

% [V,l] = gen_eig(Cyx * inv(Cxx) * Cxy, Cyy, Nfeat); % Basis in Y
V = gen_eig(Cyx * (Cxx \ Cxy), Cyy, Nfeat); % Basis in Y

U.lambda = D; 
U.basis  = U_cca;
U.V      = V;
U.method = 'CCA';
U.train  = X;
