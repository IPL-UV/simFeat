% Compute the principal components of MNF method.
% Inputs:
%       -X     : Original data. Matrix, M(samples)xN(features).
%       -Nfeat : # features extracted.
%
% Outputs:
%       -U    : Struct:
%                       -basis  : principal componets. M(samples)xNfeat(extracted features).
%                       -train  : training original data
%                       -method : feature extraction method

function U=mnf(X,Nfeat)
% MNF:Cxx*U_mnf=s*Cnn*U_mnf

N=noise(X,10);% Noise estimation
Cxx = X'*X;
Cnn = N'*N;
[U_mnf d] = gen_eig(Cxx,Cnn,Nfeat);

U.lambda=d;
U.basis=U_mnf;
U.method='MNF';
U.train=X;