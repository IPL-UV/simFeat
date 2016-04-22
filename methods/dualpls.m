function [alpha, testY, betador, varargout] = dualpls(K, Ktest, Y, T, varargin)

% function [alpha, testY, varargout] = dualpls(K, Ktest, Y, Ytest, T, varargin)
%
% Performs dual (kernel) PLS discrimination
%
% INPUTS
%  K = the training kernel matrix
%  Ktest = the kernel matrix (dimension ell x elltest)
%  Y = the training label matrix (ell x m)
%  T = the number of PLS components to take
%  varargin = optional argument specifying the true test label matrix
%             of size elltest x m, if 
%
% OUTPUTs
%  alpha = the dual vectors corresponding to the PLS classfier
%  testY = the estimated label matrix for the test samples
%  varargout = the test error, optional when varargin is specified (the
%              true test labels)
%
% For more info, see www.kernel-methods.net
%
% Note: this code has not been tested extensively.

% K is an ell x ell kernel matrix
% Y is ell x m containing the corresponding output vectors
% T gives the number of iterations to be performed
% ell = size(K,1);
% trainY = 0;
KK = K;
YY = Y;

betador = zeros(size(YY,1),T);
tau = zeros(size(KK,1),T);

for i = 1:T
    YYK = YY * YY' * KK;
    betador(:,i) = YY(:,1) / norm(YY(:,1));
    if size(YY,2) > 1, % only loop if dimension greater than 1
        bold = betador(:,i) + 1;
        contador = 0;
        while norm(betador(:,i) - bold) > 0.001,
            contador = contador + 1;
            if contador > 1000
                break
            end
            bold = betador(:,i);
            tbetador = YYK * betador(:,i);
            betador(:,i) = tbetador / norm(tbetador);
        end
    end
    tau(:,i) = KK * betador(:,i);
    val = tau(:,i)' * tau(:,i);
    %c(:,i) = YY' * tau(:,i)/val;
    c =  YY' * tau(:,i)/val;
    % trainY = trainY + tau(:,i) * c'; %c(:,i)';
    % trainerror = norm(Y - trainY,'fro')/sqrt(ell);
    w = KK * tau(:,i) / val;
    KK = KK - tau(:,i) * w' - w * tau(:,i)' + tau(:,i) * tau(:,i)' * (tau(:,i)' * w) / val;
    YY = YY - tau(:,i) * c'; %c(:,i)';
end

% Regression coefficients for new data
alpha = betador * ((tau' * K * betador) \ tau') * Y;

%  Ktest gives new data inner products as rows, Ytest true outputs
if ~isempty(Ktest)
elltest = size(Ktest',1);
testY = Ktest' * alpha;
if ~isempty(varargin)
    Ytest = varargin{1};
    testerror = norm(Ytest - testY, 'fro')/sqrt(elltest);
    varargout = testerror;
end
else
    testY = [];
end
