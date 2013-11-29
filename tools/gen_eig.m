function [U D] = gen_eig(A,B,n_eig)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extracts generalized eigenvalues for problem A * U = B * U * Landa
% n_eig -- Number of eigenvalues to compute (optional)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (nargin==3)
    n_eig = min([n_eig rank(A) rank(B)]);
else
    n_eig = min([rank(A) rank(B)]);
end
OPTS.disp = 0;

B = (B + B')/2;
R = size(B,1);
rango = rank(B);
if (rango == R)
    U = zeros(R,n_eig);
    D = zeros(n_eig,n_eig);
    inv_B = inv(B);
    for k = 1:n_eig
        [a,d] = eigs(inv_B*A,1,'LM',OPTS);
        a = a ./ sqrt(a'*B*a);
        U(:,k) = a;
        D(k,k) = d;
        A = A - d * (B * a) * (a' * B);
    end
else
%     rango = max(rango,sum(eig(B)>0) - 5);
    [v,d] = eigs(B,rango);
    B = v'*B*v;
    A = v'*A*v;
    U2 = zeros(rango,n_eig);
    D = zeros(n_eig,n_eig);
    inv_B = inv(B);
    for k = 1:n_eig
        [a,d] = eigs(inv_B*A,1,'LM',OPTS);
        a = a ./ sqrt(a'*B*a);
        U2(:,k) = a;
        D(k,k) = d;
        A = A - d * (B * a) * (a' * B);
    end
    U = v * U2;
end
