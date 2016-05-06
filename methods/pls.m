% Compute the principal components of PLS method.
% Inputs:
%       - X      : Original data. Matrix, M(samples) x N(features)
%       - Yb     : Labeled of X. Matrix, M(samples) x C(classes)
%       - Nfmax  : # features extracted
%       - method : 'PLS', as Cxy * U_pls = s * U_pls, using 'svd'
%                  'primalPLS', same but computed iteratively (see primalpls.m)
%
% Outputs:
%       - U     : Struct:
%                   - basis  : principal componets. Matrix, M(samples) x R(rank(Cxy))
%                   - train  : training original data
%                   - method : feature extraction method

function U = pls(X, Y, Nfmax, method)
% PLS: Cxy * U_pls = s * U_pls

if ~exist('method', 'var')
    method = 'PLS';
end

switch method
    case 'PLS'
        Cxy = X' * Y;
        % [A,S,V] = svds(Cxy,min(Nfmax,rank(Cxy)));
        [A,S] = svds(Cxy, min(Nfmax, rank(Cxy)));
        % [~,ind] = sort(diag(S), 'descend');
        % A = A(:,ind);
        U.labmda = S;
        
    case 'primalPLS'
        A = primalpls(X, Y, Nfmax);
        
    otherwise
        error(['Unknown method ' method])
end

U.basis = A;
U.method = method;
U.train = X;
U.Nfeat = size(A,2);
