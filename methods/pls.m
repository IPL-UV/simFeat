function [W, u, testY, testerror] = pls(X, Xtest, Y, T, varargin)

%function [W, u, testY, testerror] = pls(X, Xtest, Y, T, varargin)
%
% Performs PLS discrimination
%
%INPUTS
% X = the training kernel matrix (ell x N)
% Xtest = the sample matrix (elltest x N)
% Y = the training label matrix (ell x m)
% T = the number of PLS components to take
% varargin = optional argument specifying the true test label matrix
%            of size elltest x m
%
%OUTPUTs
% w = the weight vectors corresponding to the PLS classfier
% testY = the estimated label matrix for the test samples
% testerror = the test error
%
%
%For more info, see www.kernel-methods.net
%
%Note: this code has not been tested extensively.



% X is an ell x n matrix whose rows are the training inputs
% Y is ell x m containing the corresponding output vectors
% T gives the number of iterations to be performed


mux = mean(X); muy = mean(Y); 

trainY = 0;
% ell = size(X,1);

u = zeros(size(Y,2), T);
c = zeros(size(Y,2), T);
p = zeros(size(X,2), T);

for i = 1:T
    YX = Y' * X;
    u(:,i) = YX(1,:)' / norm(YX(1,:));
    if size(Y,2) > 1, % only loop if dimension greater than 1
        uold = u(:,i) + 1;
        contador = 0;
        while norm(u(:,i) - uold) > 0.001,
            contador = contador + 1;
            if contador > 1000
                break
            end
            uold = u(:,i);
            tu = YX' * YX * u(:,i);
            u(:,i) = tu / norm(tu);
        end
    end
    t = X * u(:,i);
    c(:,i) = Y' * t / (t'*t);
    p(:,i) = X' * t / (t'*t);
    trainY = trainY + t*c(:,i)';
    % trainerror = norm(Y - trainY,'fro') / sqrt(ell);
    X = X - t * p(:,i)';
    % compute residual Y = Y - t*c(:,i)';
end

% Regression coefficients for new data
W = u * ((p' * u) \ c'); % W = u * (inv(p' * u) * c');
%  Xtest gives new data inputs as rows, Ytest true outputs
elltest = size(Xtest,1); jj = ones(elltest,1);
testY = (Xtest - jj * mux) * W + jj * muy;

if ~isempty(varargin)
    Ytest = varargin{1};
    testerror = norm(Ytest - testY,'fro') / sqrt(elltest);
    % varargout = testerror;
end
