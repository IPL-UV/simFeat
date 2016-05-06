% Educational demo that compares standard linear and nonlinear feature extraction methods:
%       PCA, PLS, OPLS, CCA, MNF,
%       KPCA, KPLS, KOPLS, KCCA, KMNF and KECA
%
% The projected data onto the most relevant extracted features (scores)
% are used for linear classification and the overall accuracy, OA[%], and
% the estimated Cohen's kappa statistic is given. We also look at the
% robustness of the predictions as a function of the number of labels.
%
% Emma Izquierdo-Verdiguier, 2012
% emma.izquierdo@uv.es
% http://isp.uv.es

clear; clc; % close all;

fontname = 'Helvetica';
fontsize = 11;
fontunits = 'points';
linewidth = 1; % 3;
markersize = 5; % 10;

set(0,'DefaultAxesFontName',fontname,'DefaultAxesFontSize',fontsize,'DefaultAxesFontUnits',fontunits,...
    'DefaultTextFontName',fontname,'DefaultTextFontSize',fontsize,'DefaultTextFontUnits',fontunits,...
    'DefaultLineLineWidth',linewidth,'DefaultLineMarkerSize',markersize,'DefaultLineColor',[0 0 0]);

%% Path
addpath(genpath('./methods'))
addpath(genpath('./tools'))

%% Load data
% rand('seed',1234); randn('seed',1234)
rng(1234);
Ntrain = 100;   % Training data
Ntest  = 160;   % Test data
problem = 'swiss'  %  lines wheel swiss noisysinus moons xor noisyxor ellipsoids balls3 ellipsoids3
[X,Y]         = generate_toydata(Ntrain,problem);
[Xtest,Ytest] = generate_toydata(Ntest,problem);

%% Standardize the data
id = size(X,1);
Xt = [X;Xtest];
Xt = zscore(Xt);
X = Xt(1:id,:);
Xtest = Xt(1+id:end,:);

% %% Plot original distribution
% figure, plot(X(Y==1,1),X(Y==1,2),'o','MarkerFaceColor',[0 0 0.6],'MarkerEdgeColor','b','markersize',9)
% hold on, plot(X(Y==2,1),X(Y==2,2),'o','MarkerFaceColor',[0.6 0 0],'MarkerEdgeColor','r','markersize',9)
% axis off
% title('Original data')

%% Feature extraction settings
np = 35;           % Number of features to be extracted
% Yb = binarize(Y); % Encode the labels with a 1-of-C scheme
% methods = {'pca'}
% methods = {'pca' 'pls-SB' 'pls' 'opls' 'cca' 'mnf' 'kpca' 'kpls-SB' 'kpls' 'kopls' 'kcca' 'kmnf' 'keca'}; % all methods
% methods = {'pca' 'pls' 'opls' 'cca' 'mnf'}; % linear methods
methods = {'pls-SB','pls','opls','cca'}; % supervised linear methods
% methods = {'pca','mnf'}; % unsupervised linear methods
% methods = {'pls' 'opls' 'cca' 'kpls' 'kopls' 'kcca'}; % supervised
% methods = {'pca' 'mnf' 'kpca' 'kmnf' 'keca'}; % unsupervised
% methods = {'kpls-SB','kpls' 'kopls' 'kcca'}; % supervised kernel methods
% methods = {'kpca' 'kmnf' 'keca'}; % unsupervised kernel methods

% Do you want to analyze robustness to #samples by bootstrapping a 
% linear classifier working with the scores (projected data)?
linearclass = 1

% For classification labels must be encoded with a 1-of-C scheme
Yb = binarize(Y);

% For nonlinear methods, which method to use when estimating RBF 'sigma'
estimateSigmaMethod = 'mean';

%% Linear Methods

% PCA
if sum(strcmpi(methods,'pca'))
    npmax = min([np,size(X,2)]);
    U_pca = pca(X,npmax);
    Ypred_PCA = predict(Y,Xtest,U_pca,npmax);
end
% PLS-SB
if sum(strcmpi(methods,'pls-SB'))
    %npmax = min([np,size(X,2),max(Y)]);
    npmax = min([np,size(X,2),size(Yb,2)]);
    U_plsSB = plsSB(X,Yb,npmax);
    Ypred_PLSSB = predict(Y, Xtest, U_plsSB, npmax);
end
% PLS
if sum(strcmpi(methods,'pls'))
    npmax = np;
    [U_pls Ypred_PLS] = predictPLS(X, Xtest, Yb, npmax);
    %Ypred_PLS = predict(Y, Xtest, U_pls, npmax);
end
% OPLS
if sum(strcmpi(methods,'opls'))
    %npmax = min([np,size(X,2),max(Y)]);
    npmax = min([np,size(X,2),size(Yb,2)]);
    U_opls = opls(X,Yb,npmax);
    Ypred_OPLS = predict(Y,Xtest,U_opls,npmax);
end
% CCA
if sum(strcmpi(methods,'cca'))
    %npmax = min([np,size(X,2),max(Y)]);
    npmax = min([np,size(X,2),size(Yb,2)]);
    U_cca = cca(X,Yb,npmax);
    Ypred_CCA = predict(Y,Xtest,U_cca,npmax);
end
% CCA2
%if sum(strcmpi(methods,'cca2'))
%    npmax = min([np,size(X,2),max(Y)]);
%    U_cca2 = cca2(X',Y');
%    Ypred_CCA2 = predict(Y,Xtest,U_cca2,npmax);
%end
% MNF
if sum(strcmpi(methods,'mnf'))
    npmax = min([np,size(X,2)]);
    U_mnf = mnf(X,npmax);
    Ypred_MNF = predict(Y,Xtest,U_mnf,npmax);
end

%% Nonlinear methods

% KPCA
if sum(strcmpi(methods,'kpca'))
    npmax = min([np,size(X,1)]);
    U_kpca = kpca(X, npmax, estimateSigmaMethod);
    Ypred_KPCA = predict(Y, Xtest, U_kpca, npmax);
end
% KPLS-SB
if sum(strcmpi(methods,'kpls-SB'))
    %npmax = min([np,max(Y)]);
    npmax = min([np,size(Yb,2)]);
    U_kplsSB = kpls(X, Yb, npmax, estimateSigmaMethod);
    Ypred_KPLSSB = predict(Y, Xtest, U_kplsSB, npmax);
end
% KPLS
if sum(strcmpi(methods,'kpls'))
    npmax = np;
    [U_kpls Ypred_KPLS] = predictKPLS(X, Xtest, Y, npmax, estimateSigmaMethod);
end
% KOPLS
if sum(strcmpi(methods,'kopls'))
    %npmax = min([np,max(Y)]);
    npmax = min([np,size(Yb,2)]);
    U_kopls = kopls(X, Yb, npmax, estimateSigmaMethod);
    Ypred_KOPLS = predict(Y, Xtest, U_kopls, npmax);
end
% KCCA
if sum(strcmpi(methods,'kcca'))
    %npmax = min([np,max(Y)]); %size(X,1)]);
    npmax = min([np,size(Yb,2)]);
    U_kcca = kcca(X, Yb, npmax, estimateSigmaMethod);
    Ypred_KCCA = predict(Y, Xtest, U_kcca, npmax);
end
% KMNF
if sum(strcmpi(methods,'kmnf'))
    npmax = min([np,size(X,1)]);
    U_kmnf = kmnf(X, npmax, estimateSigmaMethod);
    Ypred_KMNF = predict(Y, Xtest, U_kmnf, npmax);
end
% KECA
if sum(strcmpi(methods,'keca'))
    npmax = min([np,size(X,1)]);
    U_keca = keca(X, npmax, estimateSigmaMethod);
    Ypred_KECA = predict(Y, Xtest, U_keca, npmax);
end

%% Plots

% % Principal directions
% figure,
% plot(X(Y==1,1),X(Y==1,2),'r.',X(Y==2,1),X(Y==2,2),'b.');
% hold on
% plot([0 2.5*U_pca.basis(1)],[0 2.5*U_pca.basis(2)],'k')
% plot([0 3.2*U_pls.basis(1)],[0 3.2*U_pls.basis(2)],'g')
% plot([0 3.2*U_plsSB.basis(1)],[0 3.2*U_plsSB.basis(2)],'b')
% plot([0 40*U_opls.basis(1)],[0 40*U_opls.basis(2)],'m')
% plot([0 40*U_cca.basis(1)],[0 40*U_cca.basis(2)],'c')
% plot([0 7*U_mnf.basis(1)],[0 7*U_mnf.basis(2)],'r'),axis square,axis off
% legend('Class 1','Class 2','PCA','PLS','PLS-SB','OPLS','CCA','MNF')

% Projections, Empirical mapping and classification surfaces
if sum(strcmpi(methods,'pca'))
    figures(U_pca,Y,np)
end
if sum(strcmpi(methods,'pls-SB'))
    figures(U_plsSB,Y,np)
end
if sum(strcmpi(methods,'pls'))
    figures(U_pls,Y,np)
end
if sum(strcmpi(methods,'opls'))
    figures(U_opls,Y,np)
end
if sum(strcmpi(methods,'cca'))
    figures(U_cca,Y,np)
end
%if sum(strcmpi(methods,'cca2'))
%    figures(U_cca2,Y,np)
%end
if sum(strcmpi(methods,'mnf'))
    figures(U_mnf,Y,np)
end
if sum(strcmpi(methods,'kpca'))
    figures(U_kpca,Y,np)
end
if sum(strcmpi(methods,'kpls-SB'))
    figures(U_kplsSB,Y,np)
end
if sum(strcmpi(methods,'kpls'))
    figures(U_kpls,Y,np)
end
if sum(strcmpi(methods,'kopls'))
    figures(U_kopls,Y,np)
end
if sum(strcmpi(methods,'kcca'))
    figures(U_kcca,Y,np)
end
if sum(strcmpi(methods,'kmnf'))
    figures(U_kmnf,Y,np)
end
if sum(strcmpi(methods,'keca'))
    figures(U_keca,Y,np)
end


%% Statistics
if linearclass
    
    % Accuracy vs #predictions
    [ntest d] = size(Ytest);
    
    mPCA = zeros(1,ntest);
    mMNF = zeros(1,ntest);
    mPLSSB = zeros(1,ntest);
    mPLS = zeros(1,ntest);
    mOPLS = zeros(1,ntest);
    mCCA = zeros(1,ntest);
    mKPCA = zeros(1,ntest);
    mKMNF = zeros(1,ntest);
    mKECA = zeros(1,ntest);
    mKPLSSB = zeros(1,ntest);
    mKPLS = zeros(1,ntest);
    mKOPLS = zeros(1,ntest);
    mKCCA = zeros(1,ntest);
    
    maxrep = 10;
    
    for i = 1:ntest
        
        kk = ntest - i + 1;
        kk = ceil(kk / ntest * maxrep);
        % For a fixed, same value, for all of them
        % kk = 10;
        
        %fprintf('Predicting %5d samples %5d realizations\n', i, kk)        
        
        OAvsNumPredsPCA = zeros(1, kk);
        OAvsNumPredsMNF = zeros(1, kk);
        OAvsNumPredsPLSSB = zeros(1, kk);
        OAvsNumPredsPLS = zeros(1, kk);
        OAvsNumPredsOPLS = zeros(1, kk);
        OAvsNumPredsCCA = zeros(1, kk);
        OAvsNumPredsKPCA = zeros(1, kk);
        OAvsNumPredsKMNF = zeros(1, kk);
        OAvsNumPredsKECA = zeros(1, kk);
        OAvsNumPredsKPLSSB = zeros(1, kk);
        OAvsNumPredsKPLS = zeros(1, kk);
        OAvsNumPredsKOPLS = zeros(1, kk);
        OAvsNumPredsKCCA = zeros(1, kk);
        
        for rep = 1:kk
            
            r = randperm(ntest);
        
            if sum(strcmpi(methods,'pca'))
                RES = assessment(Ytest(r(1:i)),Ypred_PCA(r(1:i)),'class');
                OAvsNumPredsPCA(rep) = RES.OA;
            end
            
            if sum(strcmpi(methods,'mnf'))
                RES = assessment(Ytest(r(1:i)),Ypred_MNF(r(1:i)),'class');
                OAvsNumPredsMNF(rep) = RES.OA;
            end
            
            if sum(strcmpi(methods,'pls-SB'))
                RES = assessment(Ytest(r(1:i)),Ypred_PLSSB(r(1:i)),'class');
                OAvsNumPredsPLSSB(rep) = RES.OA;
            end
            
            if sum(strcmpi(methods,'pls'))
                RES = assessment(Ytest(r(1:i)),Ypred_PLS(r(1:i)),'class');
                OAvsNumPredsPLS(rep) = RES.OA;
            end
            
            if sum(strcmpi(methods,'opls'))
                RES = assessment(Ytest(r(1:i)),Ypred_OPLS(r(1:i)),'class');
                OAvsNumPredsOPLS(rep) = RES.OA;
            end
            if sum(strcmpi(methods,'cca'))
                RES = assessment(Ytest(r(1:i)),Ypred_CCA(r(1:i)),'class');
                OAvsNumPredsCCA(rep) = RES.OA;
            end
            
            if sum(strcmpi(methods,'kpca'))
                RES = assessment(Ytest(r(1:i)),Ypred_KPCA(r(1:i)),'class');
                OAvsNumPredsKPCA(rep) = RES.OA;
            end
            
            if sum(strcmpi(methods,'kmnf'))
                RES = assessment(Ytest(r(1:i)),Ypred_KMNF(r(1:i)),'class');
                OAvsNumPredsKMNF(rep) = RES.OA;
            end
                        
            if sum(strcmpi(methods,'keca'))
                RES = assessment(Ytest(r(1:i)),Ypred_KECA(r(1:i)),'class');
                OAvsNumPredsKECA(rep) = RES.OA;
            end
                                    
            if sum(strcmpi(methods,'kpls-SB'))
                RES = assessment(Ytest(r(1:i)),Ypred_KPLSSB(r(1:i)),'class');
                OAvsNumPredsKPLSSB(rep) = RES.OA;
            end
            
            if sum(strcmpi(methods,'kpls'))
                RES = assessment(Ytest(r(1:i)),Ypred_KPLS(r(1:i)),'class');
                OAvsNumPredsKPLS(rep) = RES.OA;
            end
            
            if sum(strcmpi(methods,'kopls'))
                RES = assessment(Ytest(r(1:i)),Ypred_KOPLS(r(1:i)),'class');
                OAvsNumPredsKOPLS(rep) = RES.OA;
            end
            
            if sum(strcmpi(methods,'kcca'))
                RES = assessment(Ytest(r(1:i)),Ypred_KCCA(r(1:i)),'class');
                OAvsNumPredsKCCA(rep) = RES.OA;
            end
            
        end
        mPCA(i) = mean(OAvsNumPredsPCA);
        mMNF(i) = mean(OAvsNumPredsMNF);
        mPLSSB(i) = mean(OAvsNumPredsPLSSB);
        mPLS(i) = mean(OAvsNumPredsPLS);
        mOPLS(i) = mean(OAvsNumPredsOPLS);
        mCCA(i) = mean(OAvsNumPredsCCA);
        mKPCA(i) = mean(OAvsNumPredsKPCA);
        mKMNF(i) = mean(OAvsNumPredsKMNF);
        mKECA(i) = mean(OAvsNumPredsKECA);
        mKPLSSB(i) = mean(OAvsNumPredsKPLSSB);
        mKPLS(i) = mean(OAvsNumPredsKPLS);
        mKOPLS(i) = mean(OAvsNumPredsKOPLS);
        mKCCA(i) = mean(OAvsNumPredsKCCA);
    end
end

%% Figure
figure, lw = 2; ms = 10; % lw: LineWidth ms: MarkerSize
if sum(strcmpi(methods,'pca'))
    semilogx(1:ntest,(mPCA),'r-.','linewidth',lw), hold on
end
if sum(strcmpi(methods,'pls-SB'))
    semilogx(1:ntest,(mPLSSB),'b-.','Marker','.','MarkerSize',ms,'linewidth',lw), hold on
end
if sum(strcmpi(methods,'pls'))
    semilogx(1:ntest,(mPLS),'b-.','linewidth',lw), hold on
end
if sum(strcmpi(methods,'opls'))
    semilogx(1:ntest,(mOPLS),'m-.','linewidth',lw), hold on
end
if sum(strcmpi(methods,'cca'))
    semilogx(1:ntest,(mCCA),'k-.','linewidth',lw), hold on
end
%if sum(strcmpi(methods,'cca2'))
%    semilogx(1:ntest,(m(CCA2),'g-.','linewidth',lw), hold on
%end
if sum(strcmpi(methods,'mnf'))
    semilogx(1:ntest,(mMNF),'g-.','linewidth',lw), hold on
end
if sum(strcmpi(methods,'kpca'))
    semilogx(1:ntest,(mKPCA),'r','linewidth',lw), hold on
end
if sum(strcmpi(methods,'kpls-SB'))
    semilogx(1:ntest,(mKPLSSB),'b','Marker','.','MarkerSize',ms,'linewidth',lw), hold on
end
if sum(strcmpi(methods,'kpls'))
    semilogx(1:ntest,(mKPLS),'b','linewidth',lw), hold on
end
if sum(strcmpi(methods,'kopls'))
    semilogx(1:ntest,(mKOPLS),'m','linewidth',lw), hold on
end
if sum(strcmpi(methods,'kcca'))
    semilogx(1:ntest,(mKCCA),'k','linewidth',lw), hold on
end
if sum(strcmpi(methods,'kmnf'))
    semilogx(1:ntest,(mKMNF),'g','linewidth',lw), hold on
end
if sum(strcmpi(methods,'keca'))
    semilogx(1:ntest,(mKECA),'c','linewidth',lw), hold on
end
xlabel('# Predictions'), ylabel('Overall accuracy')
grid, axis tight, legend(methods)


%% Original statistics
if linearclass
    
    OAvsNumPredictionsPCA = zeros(10,ntest);
    OAvsNumPredictionsPLSSB = zeros(10,ntest);
    OAvsNumPredictionsPLS = zeros(10,ntest);
    OAvsNumPredictionsOPLS = zeros(10,ntest);
    OAvsNumPredictionsCCA = zeros(10,ntest);
    %OAvsNumPredictionsCCA2 = zeros(10,ntest);
    OAvsNumPredictionsMNF = zeros(10,ntest);
    OAvsNumPredictionsKPCA = zeros(10,ntest);
    OAvsNumPredictionsKPLSSB = zeros(10,ntest);
    OAvsNumPredictionsKPLS = zeros(10,ntest);
    OAvsNumPredictionsKOPLS = zeros(10,ntest);
    OAvsNumPredictionsKCCA = zeros(10,ntest);
    OAvsNumPredictionsKMNF = zeros(10,ntest);
    OAvsNumPredictionsKECA = zeros(10,ntest);
    
    warning('off','all')
    
    for realiza = 1:10
        r = randperm(ntest);
        %warning('off','all');
        %task = getCurrentTask;
        %fprintf('task: %d on r: %d\n', task.ID, realiza)
        for i = 1:ntest
            if sum(strcmpi(methods,'pca'))
                RES = assessment(Ytest(r(1:i)),Ypred_PCA(r(1:i)),'class');
                OAvsNumPredictionsPCA(realiza,i) = RES.OA;
            end
            if sum(strcmpi(methods,'pls-SB'))
                RES = assessment(Ytest(r(1:i)),Ypred_PLSSB(r(1:i)),'class');
                OAvsNumPredictionsPLSSB(realiza,i) = RES.OA;
            end
            if sum(strcmpi(methods,'pls'))
                RES = assessment(Ytest(r(1:i)),Ypred_PLS(r(1:i)),'class');
                OAvsNumPredictionsPLS(realiza,i) = RES.OA;
            end
            if sum(strcmpi(methods,'opls'))
                RES = assessment(Ytest(r(1:i)),Ypred_OPLS(r(1:i)),'class');
                OAvsNumPredictionsOPLS(realiza,i) = RES.OA;
            end
            if sum(strcmpi(methods,'cca'))
                RES = assessment(Ytest(r(1:i)),Ypred_CCA(r(1:i)),'class');
                OAvsNumPredictionsCCA(realiza,i) = RES.OA;
            end
            %if sum(strcmpi(methods,'cca2'))
            %    RES = assessment(Ytest(r(1:i)),Ypred_CCA2(r(1:i)),'class');
            %    OAvsNumPredictionsCCA2(realiza,i) = RES.OA;
            %end
            if sum(strcmpi(methods,'mnf'))
                RES = assessment(Ytest(r(1:i)),Ypred_MNF(r(1:i)),'class');
                OAvsNumPredictionsMNF(realiza,i) = RES.OA;
            end
            if sum(strcmpi(methods,'kpca'))
                RES = assessment(Ytest(r(1:i)),Ypred_KPCA(r(1:i)),'class');
                OAvsNumPredictionsKPCA(realiza,i) = RES.OA;
            end
            if sum(strcmpi(methods,'kpls-SB'))
                RES = assessment(Ytest(r(1:i)),Ypred_KPLSSB(r(1:i)),'class');
                OAvsNumPredictionsKPLSSB(realiza,i) = RES.OA;
            end
            if sum(strcmpi(methods,'kpls'))
                RES = assessment(Ytest(r(1:i)),Ypred_KPLS(r(1:i)),'class');
                OAvsNumPredictionsKPLS(realiza,i) = RES.OA;
            end
            if sum(strcmpi(methods,'kopls'))
                RES = assessment(Ytest(r(1:i)),Ypred_KOPLS(r(1:i)),'class');
                OAvsNumPredictionsKOPLS(realiza,i) = RES.OA;
            end
            if sum(strcmpi(methods,'kcca'))
                RES = assessment(Ytest(r(1:i)),Ypred_KCCA(r(1:i)),'class');
                OAvsNumPredictionsKCCA(realiza,i) = RES.OA;
            end
            if sum(strcmpi(methods,'kmnf'))
                RES = assessment(Ytest(r(1:i)),Ypred_KMNF(r(1:i)),'class');
                OAvsNumPredictionsKMNF(realiza,i) = RES.OA;
            end
            if sum(strcmpi(methods,'keca'))
                RES = assessment(Ytest(r(1:i)),Ypred_KECA(r(1:i)),'class');
                OAvsNumPredictionsKECA(realiza,i) = RES.OA;
            end
        end
    end
end

%% Figure
figure,
if sum(strcmpi(methods,'pca'))
    semilogx(1:ntest,mean(OAvsNumPredictionsPCA),'r-.','linewidth',lw), hold on
end
if sum(strcmpi(methods,'pls-SB'))
    semilogx(1:ntest,mean(OAvsNumPredictionsPLSSB),'b-.','Marker','.','MarkerSize',ms,'linewidth',lw), hold on
end
if sum(strcmpi(methods,'pls'))
    semilogx(1:ntest,mean(OAvsNumPredictionsPLS),'b-.','linewidth',lw), hold on
end
if sum(strcmpi(methods,'opls'))
    semilogx(1:ntest,mean(OAvsNumPredictionsOPLS),'m-.','linewidth',lw), hold on
end
if sum(strcmpi(methods,'cca'))
    semilogx(1:ntest,mean(OAvsNumPredictionsCCA),'k-.','linewidth',lw), hold on
end
%if sum(strcmpi(methods,'cca2'))
%    semilogx(1:ntest,mean(OAvsNumPredictionsCCA2),'g-.','linewidth',lw), hold on
%end
if sum(strcmpi(methods,'mnf'))
    semilogx(1:ntest,mean(OAvsNumPredictionsMNF),'g-.','linewidth',lw), hold on
end
if sum(strcmpi(methods,'kpca'))
    semilogx(1:ntest,mean(OAvsNumPredictionsKPCA),'r','linewidth',lw), hold on
end
if sum(strcmpi(methods,'kpls-SB'))
    semilogx(1:ntest,mean(OAvsNumPredictionsKPLSSB),'b','Marker','.','MarkerSize',ms,'linewidth',lw), hold on
end
if sum(strcmpi(methods,'kpls'))
    semilogx(1:ntest,mean(OAvsNumPredictionsKPLS),'b','linewidth',lw), hold on
end
if sum(strcmpi(methods,'kopls'))
    semilogx(1:ntest,mean(OAvsNumPredictionsKOPLS),'m','linewidth',lw), hold on
end
if sum(strcmpi(methods,'kcca'))
    semilogx(1:ntest,mean(OAvsNumPredictionsKCCA),'k','linewidth',lw), hold on
end
if sum(strcmpi(methods,'kmnf'))
    semilogx(1:ntest,mean(OAvsNumPredictionsKMNF),'g','linewidth',lw), hold on
end
if sum(strcmpi(methods,'keca'))
    semilogx(1:ntest,mean(OAvsNumPredictionsKECA),'c','linewidth',lw), hold on
end
xlabel('# Predictions'), ylabel('Overall accuracy')
grid, axis tight, legend(methods)
