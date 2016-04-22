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

clear;clc;close all;

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
problem = 'moons'  %  lines  wheel swiss  %%%--  noisysinus moons xor noisyxor ellipsoids balls3 ellipsoids3
[X,Y]         = generate_toydata(Ntrain,problem);
[Xtest,Ytest] = generate_toydata(Ntest,problem);

%% Standardize the data
id=size(X,1);
Xt=[X;Xtest];
Xt=scalestd(Xt);
X=Xt(1:id,:);
Xtest=Xt(1+id:end,:);

% %% Plot original distribution
% figure, plot(X(Y==1,1),X(Y==1,2),'o','MarkerFaceColor',[0 0 0.6],'MarkerEdgeColor','b','markersize',9)
% hold on, plot(X(Y==2,1),X(Y==2,2),'o','MarkerFaceColor',[0.6 0 0],'MarkerEdgeColor','r','markersize',9)
% axis off
% title('Original data')

%% Feature extraction settings
np = 35;           % Number of features to be extracted
% Yb = binarize(Y); % Encode the labels with a 1-of-C scheme
% methods = {'pca'}
methods = {'pca' 'pls-SB' 'pls' 'opls' 'cca' 'mnf' 'kpca' 'kpls-SB' 'kpls' 'kopls' 'kcca' 'kmnf' 'keca'}; % all methods
% methods = {'pca' 'pls' 'opls' 'cca' 'mnf'}; % linear methods
% methods = { 'kcca'}; % linear methods
% methods = {'pls' 'opls' 'cca' 'kpls' 'kopls' 'kcca'}; % supervised
% methods = {'pca' 'mnf' 'kpca' 'kmnf' 'keca'}; % unsupervised
% methods = {'kpls' 'kopls' 'kcca'}; % supervised kernel methods
% methods = {'kpca' 'kmnf' 'keca'}; % unsupervised kernel methods

%% Do you want to analyze robustness to #samples by bootstrapping a 
%% linear classifier working with the scores (projected data)?
linearclass = 1

%% Linear Methods

% PCA
if sum(strcmpi(methods,'pca'))
    npmax = min([np,size(X,2)]);
    U_pca = pca(X,npmax);
    Ypred_PCA = predict(Y,Xtest,U_pca,npmax);
end
% PLS-SB
if sum(strcmpi(methods,'pls-SB'))
    npmax = min([np,size(X,2),max(Y)]);
    U_plsSB = plsSB(X,Y,npmax);
    Ypred_PLSSB = predict(Y,Xtest,U_plsSB,npmax);
end
% PLS
if sum(strcmpi(methods,'pls'))
    npmax = np;
    [U_pls Ypred_PLS]=predictPLS(X,Xtest,Y,npmax);
end
% OPLS
if sum(strcmpi(methods,'opls'))
    npmax = min([np,size(X,2),max(Y)]);
    U_opls = opls(X,Y,npmax);
    Ypred_OPLS = predict(Y,Xtest,U_opls,npmax);
end
% CCA
if sum(strcmpi(methods,'cca'))
    npmax = min([np,size(X,2),max(Y)]);
    U_cca = cca(X,Y,npmax);
    Ypred_CCA = predict(Y,Xtest,U_cca,npmax);
end
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
    U_kpca = kpca(X,npmax);
    Ypred_KPCA = predict(Y,Xtest,U_kpca,npmax);
end
% KPLS-SB
if sum(strcmpi(methods,'kpls-SB'))
    npmax = min([np,max(Y)]);
    U_kplsSB = kpls(X,Y,npmax);
    Ypred_KPLSSB = predict(Y,Xtest,U_kplsSB,npmax);
end
% KPLS
if sum(strcmpi(methods,'kpls'))
    npmax = np;
    [U_kpls Ypred_KPLS]=predictKPLS(X,Xtest,Y,npmax);
end
% KOPLS
if sum(strcmpi(methods,'kopls'))
    npmax = min([np,max(Y)]);
    U_kopls = kopls(X,Y,npmax);
    Ypred_KOPLS = predict(Y,Xtest,U_kopls,npmax);
end
% KCCA
if sum(strcmpi(methods,'kcca'))
    npmax = min([np,size(X,1)]);
    U_kcca = kcca(X,Y,npmax);
    Ypred_KCCA = predict(Y,Xtest,U_kcca,npmax);
end
% KMNF
if sum(strcmpi(methods,'kmnf'))
    npmax = min([np,size(X,1)]);
    U_kmnf = kmnf(X,npmax);
    Ypred_KMNF = predict(Y,Xtest,U_kmnf,npmax);
end
% KECA
if sum(strcmpi(methods,'keca'))
    npmax = min([np,size(X,1)]);
    U_keca = keca(X,npmax);
    Ypred_KECA = predict(Y,Xtest,U_keca,npmax);
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

if linearclass
    % Accuracy vs #predictions
    [ntest d] = size(Ytest);
    warning off
    for realiza=1:10
        r=randperm(ntest);
        for i=1:ntest
            if sum(strcmpi(methods,'pca'))
                RES=assessment(Ytest(r(1:i)),Ypred_PCA(r(1:i)),'class');
                OAvsNumPredictionsPCA(realiza,i)=RES.OA;
            end
            if sum(strcmpi(methods,'pls-SB'))
                RES=assessment(Ytest(r(1:i)),Ypred_PLSSB(r(1:i)),'class');
                OAvsNumPredictionsPLSSB(realiza,i)=RES.OA;
            end
            if sum(strcmpi(methods,'pls'))
                RES=assessment(Ytest(r(1:i)),Ypred_PLS(r(1:i)),'class');
                OAvsNumPredictionsPLS(realiza,i)=RES.OA;
            end
            if sum(strcmpi(methods,'opls'))
                RES=assessment(Ytest(r(1:i)),Ypred_OPLS(r(1:i)),'class');
                OAvsNumPredictionsOPLS(realiza,i)=RES.OA;
            end
            if sum(strcmpi(methods,'cca'))
                RES=assessment(Ytest(r(1:i)),Ypred_CCA(r(1:i)),'class');
                OAvsNumPredictionsCCA(realiza,i)=RES.OA;
            end
            if sum(strcmpi(methods,'mnf'))
                RES=assessment(Ytest(r(1:i)),Ypred_MNF(r(1:i)),'class');
                OAvsNumPredictionsMNF(realiza,i)=RES.OA;
            end
            if sum(strcmpi(methods,'kpca'))
                RES=assessment(Ytest(r(1:i)),Ypred_KPCA(r(1:i)),'class');
                OAvsNumPredictionsKPCA(realiza,i)=RES.OA;
            end
            if sum(strcmpi(methods,'kpls-SB'))
                RES=assessment(Ytest(r(1:i)),Ypred_KPLSSB(r(1:i)),'class');
                OAvsNumPredictionsKPLSSB(realiza,i)=RES.OA;
            end
            if sum(strcmpi(methods,'kpls'))
                RES=assessment(Ytest(r(1:i)),Ypred_KPLS(r(1:i)),'class');
                OAvsNumPredictionsKPLS(realiza,i)=RES.OA;
            end
            if sum(strcmpi(methods,'kopls'))
                RES=assessment(Ytest(r(1:i)),Ypred_KOPLS(r(1:i)),'class');
                OAvsNumPredictionsKOPLS(realiza,i)=RES.OA;
            end
            if sum(strcmpi(methods,'kcca'))
                RES=assessment(Ytest(r(1:i)),Ypred_KCCA(r(1:i)),'class');
                OAvsNumPredictionsKCCA(realiza,i)=RES.OA;
            end
            if sum(strcmpi(methods,'kmnf'))
                RES=assessment(Ytest(r(1:i)),Ypred_KMNF(r(1:i)),'class');
                OAvsNumPredictionsKMNF(realiza,i)=RES.OA;
            end
            if sum(strcmpi(methods,'keca'))
                RES=assessment(Ytest(r(1:i)),Ypred_KECA(r(1:i)),'class');
                OAvsNumPredictionsKECA(realiza,i)=RES.OA;
            end
        end
    end
    
    figure,
    if sum(strcmpi(methods,'pca'))
        semilogx(1:ntest,mean(OAvsNumPredictionsPCA),'r-.','linewidth',3), hold on
    end
    if sum(strcmpi(methods,'pls-SB'))
        semilogx(1:ntest,mean(OAvsNumPredictionsPLSSB),'b-.','Marker','.','MarkerSize',25,'linewidth',3), hold on
    end
    if sum(strcmpi(methods,'pls'))
        semilogx(1:ntest,mean(OAvsNumPredictionsPLS),'b-.','linewidth',3), hold on
    end
    if sum(strcmpi(methods,'opls'))
        semilogx(1:ntest,mean(OAvsNumPredictionsOPLS),'m-.','linewidth',3), hold on
    end
    if sum(strcmpi(methods,'cca'))
        semilogx(1:ntest,mean(OAvsNumPredictionsCCA),'k-.','linewidth',3), hold on
    end
    if sum(strcmpi(methods,'mnf'))
        semilogx(1:ntest,mean(OAvsNumPredictionsMNF),'g-.','linewidth',3), hold on
    end
    if sum(strcmpi(methods,'kpca'))
        semilogx(1:ntest,mean(OAvsNumPredictionsKPCA),'r','linewidth',4), hold on
    end
    if sum(strcmpi(methods,'kpls-SB'))
        semilogx(1:ntest,mean(OAvsNumPredictionsKPLSSB),'b','Marker','.','MarkerSize',25,'linewidth',4), hold on
    end
    if sum(strcmpi(methods,'kpls'))
        semilogx(1:ntest,mean(OAvsNumPredictionsKPLS),'b','linewidth',4), hold on
    end
    if sum(strcmpi(methods,'kopls'))
        semilogx(1:ntest,mean(OAvsNumPredictionsKOPLS),'m','linewidth',4), hold on
    end
    if sum(strcmpi(methods,'kcca'))
        semilogx(1:ntest,mean(OAvsNumPredictionsKCCA),'k','linewidth',4), hold on
    end
    if sum(strcmpi(methods,'kmnf'))
        semilogx(1:ntest,mean(OAvsNumPredictionsKMNF),'g','linewidth',4), hold on
    end
    if sum(strcmpi(methods,'keca'))
        semilogx(1:ntest,mean(OAvsNumPredictionsKECA),'c','linewidth',4), hold on
    end
    xlabel('# Predictions')
    ylabel('Overall accuracy')
    grid
    axis tight
    legend(methods)
    
end
