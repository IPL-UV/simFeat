function figures(U,Y,Nfeat)

Nfeat = min([Nfeat,size(U.basis,2),9]);

% switch U.method
%     case {'PCA','MNF'}
%         Nfeat = min([Nfeat,size(U.train,2),9]);
%     case {'CCA','PLS','OPLS'}
%         Nfeat = min([Nfeat,size(U.train,2),max(Y),9]);
%     case {'KPCA','KMNF','KCCA','KECA'}
%         Nfeat = min([Nfeat,size(U.train,1),9]);
%     case {'KPLS','KOPLS'}
%         Nfeat = min([Nfeat,max(Y),9]);
% end

%% PROJECT DATA
if ~isfield(U,'kernel')
    XtrainProj = U.train * U.basis(:,1:Nfeat);    
else
    if strcmp(U.method,'KECA')
        XtrainProj = U.Ktrain' *  U.basis;
    else
        Kc = kernelcentering(U.Ktrain);
        XtrainProj = Kc' * U.basis(:,1:Nfeat);
    end
end

if isfield(U,'kernel') && isfield(U,'lambda')
    phi=U.basis*sqrt(U.lambda);
end

%% PLOT
if max(Y)==3
    rows = 1+ ceil(Nfeat/3);
    cols = 3;
    figure,
    subplot(rows,cols,1),
    plot(U.train(Y==1,1),U.train(Y==1,2),'o','MarkerFaceColor',[0 0 0.6],'MarkerEdgeColor','b','markersize',9), hold on,
    plot(U.train(Y==2,1),U.train(Y==2,2),'o','MarkerFaceColor',[0.6 0 0],'MarkerEdgeColor','r','markersize',9), hold on,
    plot(U.train(Y==3,1),U.train(Y==3,2),'o','MarkerFaceColor',[0 0.6 0],'MarkerEdgeColor','g','markersize',9)
    axis equal off
    title('Original data')
    if size(XtrainProj,2)>1
        subplot(rows,cols,2),
        plot(XtrainProj(Y==1,1),XtrainProj(Y==1,2),'o','MarkerFaceColor',[0 0 0.6],'MarkerEdgeColor','b','markersize',9)
        hold on, plot(XtrainProj(Y==2,1),XtrainProj(Y==2,2),'o','MarkerFaceColor',[0.6 0 0],'MarkerEdgeColor','r','markersize',9)
        hold on, plot(XtrainProj(Y==3,1),XtrainProj(Y==3,2),'o','MarkerFaceColor',[0 0.6 0],'MarkerEdgeColor','g','markersize',9)
        axis equal off
        title(strcat(U.method,' Scores'))
        subplot(rows,cols,3),
        if isfield(U,'kernel') && isfield(U,'lambda')
            plot(phi(Y==1,1),phi(Y==1,2),'o','MarkerFaceColor',[0 0 0.6],'MarkerEdgeColor','b','markersize',9)
            hold on, plot(phi(Y==2,1),phi(Y==2,2),'o','MarkerFaceColor',[0.6 0 0],'MarkerEdgeColor','r','markersize',9)
            hold on, plot(phi(Y==3,1),phi(Y==3,2),'o','MarkerFaceColor',[0 0.6 0],'MarkerEdgeColor','g','markersize',9)
            axis equal off
            title(strcat(U.method,' Empirical Mapping'))
        else
            axis equal off
        end
    end
    for f=1:Nfeat
        subplot(rows,cols,f+3),
        if ~isfield(U,'kernel')
            plotFeatures(U.train,U.basis,Y,f);
        else
            plotKernelFeatures(U.train,U.sigma,U.basis,Y,U.Ktrain,U.method,f);
        end
        if isfield(U,'lambda')
            format='%0.4g';
            title([U.method ', \lambda=' num2str(U.lambda(f,f),format)])
        end
    end
else
    rows = 1+ ceil(Nfeat/3);
    cols = 3;
    figure,
    subplot(rows,cols,1),
    plot(U.train(Y==1,1),U.train(Y==1,2),'o','MarkerFaceColor',[0 0 0.6],'MarkerEdgeColor','b','markersize',9), hold on,
    plot(U.train(Y==2,1),U.train(Y==2,2),'o','MarkerFaceColor',[0.6 0 0],'MarkerEdgeColor','r','markersize',9)
    axis equal off
    title('Original data')
    if size(XtrainProj,2)>1
        subplot(rows,cols,2),
        plot(XtrainProj(Y==1,1),XtrainProj(Y==1,2),'o','MarkerFaceColor',[0 0 0.6],'MarkerEdgeColor','b','markersize',9)
        hold on, plot(XtrainProj(Y==2,1),XtrainProj(Y==2,2),'o','MarkerFaceColor',[0.6 0 0],'MarkerEdgeColor','r','markersize',9)
        axis equal off
        title(strcat(U.method,' Scores'))
        subplot(rows,cols,3),
        if isfield(U,'kernel') && isfield(U,'lambda')
            plot(phi(Y==1,1),phi(Y==1,2),'o','MarkerFaceColor',[0 0 0.6],'MarkerEdgeColor','b','markersize',9)
            hold on, plot(phi(Y==2,1),phi(Y==2,2),'o','MarkerFaceColor',[0.6 0 0],'MarkerEdgeColor','r','markersize',9)
            axis equal off
            title(strcat(U.method,' Empirical Mapping'))
        else
            axis equal off
        end
    end
    for f=1:Nfeat
        subplot(rows,cols,f+3),
        if ~isfield(U,'kernel')
            plotFeatures(U.train,U.basis,Y,f);
        else
            plotKernelFeatures(U.train,U.sigma,U.basis,Y,U.Ktrain,U.method,f);
        end
        if isfield(U,'lambda')
            format='%0.4g';
            title([U.method ', \lambda=' num2str(U.lambda(f,f),format)])
        end
    end
end
set(gcf,'DoubleBuffer','on');

% for i = 1:Nfeatures
%     F=get_gabor(sz,Angulos(i),Frecs(i),Sigmas(i,:),Posiciones(i,:));
%     Filtros(:,i) = F(:);
% end
% Filtros(:,Nfeatures+1:ceil(sqrt(Nfeatures))^2) = 0;
% figure,
% imagesc(disp_patches(Filtros,ceil(sqrt(Nfeatures)))),title('Gabor features'),colormap gray

% if ~isfield(U,'kernel')
%     
%     XtrainProj = U.train * U.basis(:,1:Nfeat);
%     figure,
%     plotFeatures(U.train,U.basis,Y,U.method,Nfeat)
%     
% else
%     if strcmp(U.method,'KECA')
%         XtrainProj = U.Ktrain' *  U.basis;
%     else
%         Kc=kernelcentering(U.Ktrain);
%         XtrainProj=Kc' * U.basis(:,1:Nfeat);
%     end
%     
%     phi=U.basis*sqrt(U.lambda);
%     
%     figure, plot(phi(Y==1,1),phi(Y==1,2),'o','MarkerFaceColor',[0 0 0.6],'MarkerEdgeColor','b','markersize',9)
%     hold on, plot(phi(Y==2,1),phi(Y==2,2),'o','MarkerFaceColor',[0.6 0 0],'MarkerEdgeColor','r','markersize',9)
%     axis off
%     title(strcat(U.method,' Empirical Mapping'))
%     
%     figure, plotKernelFeatures(U.train,U.sigma,U.basis,Y,U.Ktrain,U.method,Nfeat)
% end
% 
% figure, plot(XtrainProj(Y==1,1),XtrainProj(Y==1,2),'o','MarkerFaceColor',[0 0 0.6],'MarkerEdgeColor','b','markersize',9)
% hold on, plot(XtrainProj(Y==2,1),XtrainProj(Y==2,2),'o','MarkerFaceColor',[0.6 0 0],'MarkerEdgeColor','r','markersize',9)
% axis off
% title(strcat(U.method,' scores'))
