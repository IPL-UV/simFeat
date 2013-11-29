function figures(U,Y,Nfeat)

if ~isfield(U,'kernel')
    
    XtrainProj = U.train * U.basis(:,1:Nfeat);
    figure, 
    plotFeatures(U.train,U.basis,Y,U.method,Nfeat)
    
else
    if strcmp(U.method,'KECA')
        XtrainProj = U.Ktrain' *  U.basis;
    else
        Kc=kernelcentering(U.Ktrain);
        XtrainProj=Kc' * U.basis(:,1:Nfeat);
    end
    
    phi=U.basis*sqrt(U.lambda);
    
    figure, plot(phi(Y==1,1),phi(Y==1,2),'o','MarkerFaceColor',[0 0 0.6],'MarkerEdgeColor','b','markersize',9)
    hold on, plot(phi(Y==2,1),phi(Y==2,2),'o','MarkerFaceColor',[0.6 0 0],'MarkerEdgeColor','r','markersize',9)
    axis off
    title(strcat(U.method,' Empirical Mapping'))
    
    figure, plotKernelFeatures(U.train,U.sigma,U.basis,Y,U.Ktrain,U.method,Nfeat)
end

figure, plot(XtrainProj(Y==1,1),XtrainProj(Y==1,2),'o','MarkerFaceColor',[0 0 0.6],'MarkerEdgeColor','b','markersize',9)
hold on, plot(XtrainProj(Y==2,1),XtrainProj(Y==2,2),'o','MarkerFaceColor',[0.6 0 0],'MarkerEdgeColor','r','markersize',9)
axis off
title(strcat(U.method,' scores'))
