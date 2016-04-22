% Estimates the sigma parameter (RBF kernel lengthscale) from available data.


% ESTIMATESIGMA This function estimates the Sigma from available data.
%
%    sigma = estimateSigma(X1,X2,Y)
%
%    INPUTS:
%       X1 : data view 1, n samples, F features: n x F
%       X2 : data view 2, n samples, F features: n x F
%       Y  : labels for data, n x 1 (optional)
%
%    OUTPUTS:
%       sigma: the estimated sigma
%
%    NOTES:
%       - Samples in X1 and X2 are two different "views" of the same data!
%       - The sigma is estimated with the median distance between all data
%       - For the 'class' option, the mean of the sigma per class is returned
%
%    USES:
%             sigma = estimateSigma(X1,X1)
%                  It estimates the sigma without class info for the kernel K(X1,X1)
%
%             sigma = estimateSigma(X1,X2,Y)
%                  It estimates the sigma with class info for the kernel K(X1,X2)
%
% Gustavo Camps-Valls, 2009
% gcamps@uv.es
%

function sigma = estimateSigma(X1,X2,Y)

if nargin < 3
    method = 'noclass';
end   

if strcmp(method,'noclass')

    samples = size(X1,1);
    G = sum((X1.*X2),2);
    Q = repmat(G,1,samples);
    R = repmat(G',samples,1);
    dists = Q + R - 2*X1*X2';
    dists = dists-tril(dists);
    dists=reshape(dists,samples^2,1);
    sigma = sqrt(0.5*mean(dists(dists>0)));

else
    Classes = unique(Y);
    NumClasses = length(Classes);
    S = zeros(1,NumClasses);
    for c = 1:NumClasses
        % Take samples of each class
        % idx = find(Y==Classees(c));
        % XX1 = X1(idx,:);
        % XX2 = X2(idx,:);
        % Compute the median distance
        samples = size(X1,1);
        G = sum((X1 .* X2),2);
        Q = repmat(G,1,samples);
        R = repmat(G',samples,1);
        dists = Q + R - 2 * X1 * X2';
        dists = dists - tril(dists);
        dists = reshape(dists,samples^2,1);
        S(c) = sqrt(0.5*mean(dists(dists>0)));
    end
    sigma = mean(S);
end
