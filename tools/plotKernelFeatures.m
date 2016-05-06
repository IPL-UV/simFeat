% Kernel principal components

function plotKernelFeatures(Xtrain,sigma,V,Labels,Ktrain,method,no_feats)

V = real(V); %% hack!
colors  = {'b' 'r' 'g' 'y' 'k' 'm' 'c'};
colors2 = {[0 0 0.6],[0.6 0 0],[0 0.6 0],[0.6 0.6 0],[0.6 0.6 0.6],[0.6 0 0.6],[0 0.6 0.6]};

gr = 30;
x1min = 2 * min(Xtrain(:,1));  x1max = 2 * max(Xtrain(:,1));
x2min = 2 * min(Xtrain(:,2));  x2max = 2 * max(Xtrain(:,2));
[X,Y] = meshgrid(linspace(x1min,x1max,gr),linspace(x2min,x2max,gr));
Z = [X(:) , Y(:)];

Ktest = kernel('rbf', Xtrain, Z, sigma);

if ~strcmp(method,'KECA')
    Ktest=kernelcentering(Ktest, sum(Ktrain));
end

phiZ = Ktest' * V;

% Plot the map
for class = 1:min(8,max(Labels))
    c = find(Labels == class);
    plot(Xtrain(c,1),Xtrain(c,2),'o','MarkerFaceColor',colors2{class},'MarkerEdgeColor',colors{class},'markersize',9),
    hold on
end

hold on
pcolor(X, Y, reshape(phiZ(:,no_feats), [gr gr])), colormap gray
% hold on
contour(X, Y, reshape(phiZ(:,no_feats), [gr gr]), 8, 'w')
shading interp
axis equal off
