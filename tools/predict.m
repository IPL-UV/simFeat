% Linear classification using the data projected onto the most relevant 
% extracted features
%
% Inputs:
%        Xtest : Test data.  Matrix, N(samples) x F(features)
%        U     : Projection model:
%                  - basis  : principal components
%                  - train  : training original data
%                  - method : feature extraction method
%                  - kernel : kernel type (e.g. 'rbf', 'lin', 'poly').
%                  - Ktrain : train kernel (for nonlinear methods)
%        Nfeat : # features to be extracted
%
% Outputs:
%        Ypred : Predicted labels. Vector, 1 x F(features)

function Ypred = predict(Y, Xtest, U, Nfeat)

Nfeat = min([Nfeat,size(U.basis,2),9]);
% switch U.method
%     case {'PCA','MNF'}
%         Nfeat = min([Nfeat,size(U.train,2)]);
%     case {'CCA','PLS','OPLS'}
%         Nfeat = min([Nfeat,size(U.train,2),max(Y)]);
%     case {'KPCA','KMNF','KCCA','KECA'}
%         Nfeat = min([Nfeat,size(U.train,1)]);
%     case {'KPLS','KOPLS'}
%         Nfeat = min([Nfeat,max(Y)]);
% end

if ~isfield(U,'kernel')    
    % Projected original data
    XtrainProj = U.train * U.basis(:,1:Nfeat);
    XtestProj = Xtest * U.basis(:,1:Nfeat);
elseif strcmp(U.method,'KECA')
    Ktest = (1/sqrt(2*U.sigma*pi)) * kernel(U.kernel,U.train,Xtest,U.sigma);
    XtrainProj = U.Ktrain' * U.basis(:,1:Nfeat);
    XtestProj = Ktest' * U.basis(:,1:Nfeat);
else
    Kc = kernelcentering(U.Ktrain);
    Ktest = kernel(U.kernel,U.train,Xtest,U.sigma);
    Kctest = kernelcentering(Ktest,sum(U.Ktrain));
    % Projected original data
    XtrainProj = Kc' * U.basis(:,1:Nfeat);
    XtestProj = Kctest' * U.basis(:,1:Nfeat);
end

% figure, plot(XtrainProj(Y==1,1),XtrainProj(Y==1,2),'o','MarkerFaceColor',[0 0 0.6],'MarkerEdgeColor','b','markersize',9)
% hold on, plot(XtrainProj(Y==2,1),XtrainProj(Y==2,2),'o','MarkerFaceColor',[0.6 0 0],'MarkerEdgeColor','r','markersize',9)
% axis off
% title(strcat(U.method,' scores'))

% Prediction labels using linear regression as basic classifier
Yb = binarize(Y);
XtrainProj1 = [XtrainProj ones(size(XtrainProj,1),1)];
W = pinv(XtrainProj1) * Yb;
Ypred = XtestProj * W(1:end-1,:) + repmat(W(end,:),size(XtestProj,1),1);
[~, Ypred] = max(Ypred,[],2);
