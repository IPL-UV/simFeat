function U = primalpls(X, Y, Nfeat)

% function [W, u, testY, testerror] = primalpls(X, Y, Nfeat)
%
% Inputs:
%       - X     : Original data. Matrix, M(samples) x N(features)
%       - Y     : Labeled of X. Vector, M(samples) x 1
%       - Nfmax : # features extracted
%
% Outputs:
%       - U     : principal componets. Matrix, M(samples) x R(rank(Cxy))

% For more info, see www.kernel-methods.net
%
% Note: this code has not been tested extensively

% X is an ell x n matrix whose rows are the training inputs
% Y is ell x m containing the corresponding output vectors
% Nfeat gives the number of iterations to be performed

Nfeat = min([Nfeat, size(X,2), size(Y,2)]);

U = zeros(size(X,2), Nfeat);
p = zeros(size(X,2), Nfeat);

for i = 1:Nfeat
    YX = Y' * X;
    U(:,i) = YX(1,:)' / norm(YX(1,:));
    if size(Y,2) > 1, % only loop if dimension greater than 1
        uold = U(:,i) + 1;
        contador = 0;
        while norm(U(:,i) - uold) > 0.001,
            contador = contador + 1;
            if contador > 1000
                break
            end
            uold = U(:,i);
            tu = YX' * YX * U(:,i);
            U(:,i) = tu / norm(tu);
        end
    end
    t = X * U(:,i);
    p(:,i) = X' * t / (t'*t);
    X = X - t * p(:,i)';
end
