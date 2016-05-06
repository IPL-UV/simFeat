function [X,Y] = loadDataset(name)

dtpath = 'Data/Classification/';

switch name
    
    case 'IndianPines'
        
        load([dtpath name]);
        X = reshape(Xtotal, [], size(Xtotal,3));
        Y = reshape(Ytotal, [], 1);
        
    case { 'pima-indians-diabetes', 'wdbc' }
        
        load([dtpath name]);
        X = Xtrain;
        Y = Ytrain;
        
    case 'ionosphere'
        
        load([dtpath name]);
        X = Xtrain(:, [1 3:end]);
        Y = Ytrain;
        
    case 'letter'
        
        load([dtpath name]);
        X = [ Xtrain ; Xtest ];
        Y = [ Ytrain ; Ytest ];
        
    otherwise
        error(['Unknown dataset ' name])
end
