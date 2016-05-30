% Educational demo that compares standard linear and nonlinear feature extraction methods:
%
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

% Ntrain = 100;   % Training data
% Ntest  = 160;   % Test data
% dataset = 'moons'  %  lines wheel swiss noisysinus moons xor noisyxor ellipsoids balls3 ellipsoids3
% [X,Y]         = generate_toydata(Ntrain, dataset);
% [Xtest,Ytest] = generate_toydata(Ntest, dataset);

% Classification example using (real) data
dataset = 'ionosphere'; % ionosphere pima-indians-diabetes wdbc letter IndianPines
[X,Y] = loadDataset(dataset);
Ntrain = fix( size(X,1) * 1/3 );
r = randperm(size(X,1));
Xtest = X(r(Ntrain+1:end), [1 3:end]);
Ytest = Y(r(Ntrain+1:end), :);
X = X(r(1:Ntrain), [1 3:end]);
Y = Y(r(1:Ntrain), :);


%% Standardize the data
id = size(X,1);
Xt = [X ; Xtest];
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
% methods = {'pca' 'pls' 'primalpls' 'opls' 'cca' 'mnf' 'kpca' 'kpls' 'dualpls' 'kopls' 'kcca' 'kmnf' 'keca'}; % all methods
% methods = {'pca' 'pls' 'opls' 'cca' 'mnf'}; % linear methods
% methods = {'pls','primalpls','opls','cca'}; % supervised linear methods
% methods = {'pca','mnf'}; % unsupervised linear methods
% methods = {'pls' 'opls' 'cca' 'kpls' 'kopls' 'kcca'}; % supervised
% methods = {'pca' 'mnf' 'kpca' 'kmnf' 'keca'}; % unsupervised
% methods = {'kpls','dualpls' 'kopls' 'kcca'}; % supervised kernel methods
% methods = {'kpca' 'kmnf' 'keca'}; % unsupervised kernel methods

methods = {'pls','primalpls','kpls','dualpls'};

% Do you want to analyze robustness to #samples by bootstrapping a
% linear classifier working with the scores (projected data)?
linearclass = 1;

% For classification labels must be encoded with a 1-of-C scheme
Yb = binarize(Y);

% For nonlinear methods, which method to use when estimating RBF 'sigma'
estimateSigmaMethod = 'mean';

%% Linear Methods

% PCA
if sum(strcmpi(methods, 'pca'))
    npmax = min([np, size(X,2)]);
    U_pca = pca(X, npmax);
    Ypred_PCA = predict(Y, Xtest, U_pca, npmax);
end
% PLS
if sum(strcmpi(methods, 'pls'))
    npmax = min([np, size(X,2), size(Yb,2)]);
    U_pls = pls(X, Yb, npmax , 'PLS');
    Ypred_PLS = predict(Y, Xtest, U_pls, npmax);
end
% Primal PLS
if sum(strcmpi(methods, 'primalpls'))
    npmax = np;
    U_ppls = pls(X, Yb, npmax, 'primalPLS');
    Ypred_PPLS = predict(Y, Xtest, U_ppls, npmax);
end
% OPLS
if sum(strcmpi(methods, 'opls'))
    npmax = min([np, size(X,2), size(Yb,2)]);
    U_opls = opls(X, Yb, npmax);
    Ypred_OPLS = predict(Y, Xtest, U_opls, npmax);
end
% CCA
if sum(strcmpi(methods, 'cca'))
    npmax = min([np, size(X,2), size(Yb,2)]);
    U_cca = cca(X, Yb, npmax);
    Ypred_CCA = predict(Y, Xtest, U_cca, npmax);
end
% CCA2
%if sum(strcmpi(methods, 'cca2'))
%    npmax = min([np, size(X,2), max(Y)]);
%    U_cca2 = cca2(X',Y');
%    Ypred_CCA2 = predict(Y, Xtest, U_cca2, npmax);
%end
% MNF
if sum(strcmpi(methods, 'mnf'))
    npmax = min([np, size(X,2)]);
    U_mnf = mnf(X, npmax);
    Ypred_MNF = predict(Y, Xtest, U_mnf, npmax);
end

%% Nonlinear methods

% KPCA
if sum(strcmpi(methods, 'kpca'))
    npmax = min([np, size(X,1)]);
    U_kpca = kpca(X, npmax, estimateSigmaMethod);
    Ypred_KPCA = predict(Y, Xtest, U_kpca, npmax);
end
% KPLS
if sum(strcmpi(methods,  'kpls'))
    npmax = min([np, size(Yb,2)]);
    U_kpls = kpls(X, Yb, npmax, 'KPLS', estimateSigmaMethod);
    Ypred_KPLS = predict(Y, Xtest, U_kpls, npmax);
end
% KPLS
if sum(strcmpi(methods, 'dualpls'))
    npmax = np;
    U_dkpls = kpls(X, Yb, npmax, 'dualPLS', estimateSigmaMethod);
    Ypred_DKPLS = predict(Y, Xtest, U_dkpls, npmax);
end
% KOPLS
if sum(strcmpi(methods, 'kopls'))
    npmax = min([np, size(Yb,2)]);
    U_kopls = kopls(X, Yb, npmax, estimateSigmaMethod);
    Ypred_KOPLS = predict(Y, Xtest, U_kopls, npmax);
end
% KCCA
if sum(strcmpi(methods, 'kcca'))
    npmax = min([np, size(Yb,2)]);
    U_kcca = kcca(X, Yb, npmax, estimateSigmaMethod);
    Ypred_KCCA = predict(Y, Xtest, U_kcca, npmax);
end
% KMNF
if sum(strcmpi(methods, 'kmnf'))
    npmax = min([np, size(X,1)]);
    U_kmnf = kmnf(X, npmax, estimateSigmaMethod);
    Ypred_KMNF = predict(Y, Xtest, U_kmnf, npmax);
end
% KECA
if sum(strcmpi(methods, 'keca'))
    npmax = min([np, size(X,1)]);
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
% plot([0 3.2*U_ppls.basis(1)],[0 3.2*U_ppls.basis(2)],'b')
% plot([0 40*U_opls.basis(1)],[0 40*U_opls.basis(2)],'m')
% plot([0 40*U_cca.basis(1)],[0 40*U_cca.basis(2)],'c')
% plot([0 7*U_mnf.basis(1)],[0 7*U_mnf.basis(2)],'r'),axis square,axis off
% legend('Class 1','Class 2','PCA','PLS','primalPLS','OPLS','CCA','MNF')

% Projections, Empirical mapping and classification surfaces

if size(X,2) < 3
    if sum(strcmpi(methods, 'pca'))
        figures(U_pca,Y,np)
    end
    if sum(strcmpi(methods, 'pls'))
        figures(U_pls,Y,np)
    end
    if sum(strcmpi(methods, 'primalpls'))
        figures(U_ppls,Y,np)
    end
    if sum(strcmpi(methods, 'opls'))
        figures(U_opls,Y,np)
    end
    if sum(strcmpi(methods, 'cca'))
        figures(U_cca,Y,np)
    end
    %if sum(strcmpi(methods, 'cca2'))
    %    figures(U_cca2,Y,np)
    %end
    if sum(strcmpi(methods, 'mnf'))
        figures(U_mnf,Y,np)
    end
    if sum(strcmpi(methods, 'kpca'))
        figures(U_kpca,Y,np)
    end
    if sum(strcmpi(methods, 'kpls'))
        figures(U_kpls,Y,np)
    end
    if sum(strcmpi(methods, 'dualpls'))
        figures(U_dkpls,Y,np)
    end
    if sum(strcmpi(methods, 'kopls'))
        figures(U_kopls,Y,np)
    end
    if sum(strcmpi(methods, 'kcca'))
        figures(U_kcca,Y,np)
    end
    if sum(strcmpi(methods, 'kmnf'))
        figures(U_kmnf,Y,np)
    end
    if sum(strcmpi(methods, 'keca'))
        figures(U_keca,Y,np)
    end
end

%% Statistics
if linearclass
    
    % Accuracy vs #predictions
    [ntest d] = size(Ytest);
    
    mPCA = zeros(1,ntest);
    mMNF = zeros(1,ntest);
    mPLS = zeros(1,ntest);
    mPPLS = zeros(1,ntest);
    mOPLS = zeros(1,ntest);
    mCCA = zeros(1,ntest);
    mKPCA = zeros(1,ntest);
    mKMNF = zeros(1,ntest);
    mKECA = zeros(1,ntest);
    mKPLS = zeros(1,ntest);
    mDKPLS = zeros(1,ntest);
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
        OAvsNumPredsPLS = zeros(1, kk);
        OAvsNumPredsPPLS = zeros(1, kk);
        OAvsNumPredsOPLS = zeros(1, kk);
        OAvsNumPredsCCA = zeros(1, kk);
        OAvsNumPredsKPCA = zeros(1, kk);
        OAvsNumPredsKMNF = zeros(1, kk);
        OAvsNumPredsKECA = zeros(1, kk);
        OAvsNumPredsKPLS = zeros(1, kk);
        OAvsNumPredsDKPLS = zeros(1, kk);
        OAvsNumPredsKOPLS = zeros(1, kk);
        OAvsNumPredsKCCA = zeros(1, kk);
        
        for rep = 1:kk
            
            r = randperm(ntest);
            
            if sum(strcmpi(methods, 'pca'))
                RES = assessment(Ytest(r(1:i)),Ypred_PCA(r(1:i)),'class');
                OAvsNumPredsPCA(rep) = RES.OA;
            end
            
            if sum(strcmpi(methods, 'mnf'))
                RES = assessment(Ytest(r(1:i)),Ypred_MNF(r(1:i)),'class');
                OAvsNumPredsMNF(rep) = RES.OA;
            end
            
            if sum(strcmpi(methods, 'pls'))
                RES = assessment(Ytest(r(1:i)),Ypred_PLS(r(1:i)),'class');
                OAvsNumPredsPLS(rep) = RES.OA;
            end
            
            if sum(strcmpi(methods, 'primalpls'))
                RES = assessment(Ytest(r(1:i)),Ypred_PPLS(r(1:i)),'class');
                OAvsNumPredsPPLS(rep) = RES.OA;
            end
            
            if sum(strcmpi(methods, 'opls'))
                RES = assessment(Ytest(r(1:i)),Ypred_OPLS(r(1:i)),'class');
                OAvsNumPredsOPLS(rep) = RES.OA;
            end
            if sum(strcmpi(methods, 'cca'))
                RES = assessment(Ytest(r(1:i)),Ypred_CCA(r(1:i)),'class');
                OAvsNumPredsCCA(rep) = RES.OA;
            end
            
            if sum(strcmpi(methods, 'kpca'))
                RES = assessment(Ytest(r(1:i)),Ypred_KPCA(r(1:i)),'class');
                OAvsNumPredsKPCA(rep) = RES.OA;
            end
            
            if sum(strcmpi(methods, 'kmnf'))
                RES = assessment(Ytest(r(1:i)),Ypred_KMNF(r(1:i)),'class');
                OAvsNumPredsKMNF(rep) = RES.OA;
            end
            
            if sum(strcmpi(methods, 'keca'))
                RES = assessment(Ytest(r(1:i)),Ypred_KECA(r(1:i)),'class');
                OAvsNumPredsKECA(rep) = RES.OA;
            end
            
            if sum(strcmpi(methods, 'kpls'))
                RES = assessment(Ytest(r(1:i)),Ypred_KPLS(r(1:i)),'class');
                OAvsNumPredsKPLS(rep) = RES.OA;
            end
            
            if sum(strcmpi(methods, 'dualpls'))
                RES = assessment(Ytest(r(1:i)),Ypred_DKPLS(r(1:i)),'class');
                OAvsNumPredsDKPLS(rep) = RES.OA;
            end
            
            if sum(strcmpi(methods, 'kopls'))
                RES = assessment(Ytest(r(1:i)),Ypred_KOPLS(r(1:i)),'class');
                OAvsNumPredsKOPLS(rep) = RES.OA;
            end
            
            if sum(strcmpi(methods, 'kcca'))
                RES = assessment(Ytest(r(1:i)),Ypred_KCCA(r(1:i)),'class');
                OAvsNumPredsKCCA(rep) = RES.OA;
            end
            
        end
        mPCA(i) = mean(OAvsNumPredsPCA);
        mMNF(i) = mean(OAvsNumPredsMNF);
        mPLS(i) = mean(OAvsNumPredsPLS);
        mPPLS(i) = mean(OAvsNumPredsPPLS);
        mOPLS(i) = mean(OAvsNumPredsOPLS);
        mCCA(i) = mean(OAvsNumPredsCCA);
        mKPCA(i) = mean(OAvsNumPredsKPCA);
        mKMNF(i) = mean(OAvsNumPredsKMNF);
        mKECA(i) = mean(OAvsNumPredsKECA);
        mKPLS(i) = mean(OAvsNumPredsKPLS);
        mDKPLS(i) = mean(OAvsNumPredsDKPLS);
        mKOPLS(i) = mean(OAvsNumPredsKOPLS);
        mKCCA(i) = mean(OAvsNumPredsKCCA);
    end
end

%% Figure
figure, lw = 2; ms = 10; % lw: LineWidth ms: MarkerSize
if sum(strcmpi(methods, 'pca'))
    semilogx(1:ntest,(mPCA),'r-.','linewidth',lw), hold on
end
if sum(strcmpi(methods, 'pls'))
    semilogx(1:ntest,(mPLS),'b-.','Marker','.','MarkerSize',ms,'linewidth',lw), hold on
end
if sum(strcmpi(methods, 'primalpls'))
    semilogx(1:ntest,(mPPLS),'c-.','linewidth',lw), hold on
end
if sum(strcmpi(methods, 'opls'))
    semilogx(1:ntest,(mOPLS),'m-.','linewidth',lw), hold on
end
if sum(strcmpi(methods, 'cca'))
    semilogx(1:ntest,(mCCA),'k-.','linewidth',lw), hold on
end
%if sum(strcmpi(methods, 'cca2'))
%    semilogx(1:ntest,(m(CCA2),'g-.','linewidth',lw), hold on
%end
if sum(strcmpi(methods, 'mnf'))
    semilogx(1:ntest,(mMNF),'g-.','linewidth',lw), hold on
end
if sum(strcmpi(methods, 'kpca'))
    semilogx(1:ntest,(mKPCA),'r','linewidth',lw), hold on
end
if sum(strcmpi(methods, 'kpls'))
    semilogx(1:ntest,(mKPLS),'b','Marker','.','MarkerSize',ms,'linewidth',lw), hold on
end
if sum(strcmpi(methods, 'dualpls'))
    semilogx(1:ntest,(mDKPLS),'c','linewidth',lw), hold on
end
if sum(strcmpi(methods, 'kopls'))
    semilogx(1:ntest,(mKOPLS),'m','linewidth',lw), hold on
end
if sum(strcmpi(methods, 'kcca'))
    semilogx(1:ntest,(mKCCA),'k','linewidth',lw), hold on
end
if sum(strcmpi(methods, 'kmnf'))
    semilogx(1:ntest,(mKMNF),'g','linewidth',lw), hold on
end
if sum(strcmpi(methods, 'keca'))
    semilogx(1:ntest,(mKECA),'c','linewidth',lw), hold on
end
xlabel('# Predictions'), ylabel('Overall accuracy')
grid, axis tight, legend(methods), title(dataset)

