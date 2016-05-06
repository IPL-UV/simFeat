function U = dualpls(K, Y, Nfeat)

% function U = dualpls(K, Y, Nfeat)
%
% Performs dual (kernel) PLS discrimination
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

% K is an ell x ell kernel matrix
% Y is ell x m containing the corresponding output vectors
% Nfeat gives the number of iterations to be performed

Nfeat = min([Nfeat, size(K,1), size(Y,2)]);

U = zeros(size(Y,1),Nfeat);
tau = zeros(size(K,1),Nfeat);

for i = 1:Nfeat
    
    YK = Y * Y' * K;
    U(:,i) = Y(:,1) / norm(Y(:,1));
    
    bold = U(:,i) + 1;
    contador = 0;
    while norm(U(:,i) - bold) > 0.001 && contador < 1000,
        bold = U(:,i);
        tU = YK * U(:,i);
        U(:,i) = tU / norm(tU);
        contador = contador + 1;
    end
    
    if i == Nfeat, break, end
    
    tau(:,i) = K * U(:,i);
    val = tau(:,i)' * tau(:,i);
    c =  Y' * tau(:,i)/val;
    w = K * tau(:,i) / val;
    K = K - tau(:,i) * w' - w * tau(:,i)' + tau(:,i) * tau(:,i)' * (tau(:,i)' * w) / val;
    Y = Y - tau(:,i) * c';
end
