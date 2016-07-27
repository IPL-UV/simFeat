SIMFEAT TOOLBOX 2.2

AUTHORS: E. Izquierdo-Verdiguier, L. Gómez-Chova, G. Camps-Valls, Jordi Muñoz-Marí

  Note:
  -----
  This toolbox requires the Alexander Ihler's kernel density estimation (KDE) toolbox. You should
  download it from http://www.ics.uci.edu/~ihler/code/kde.html and install it inside the 'tools'
  directory.

 %%% FILES %%%
 
  binarize        - Binarize of a vector.
  cca             - Compute the principal components of PLS method.
  estimateSigma   - Estimates the sigma parameter (RBF kernel lengthscale) from available data.
  gen_eig         - Extracts generalized eigenvalues for problem A * U = B * U * Landa.
  kcca            - Compute the principal components of KCCA method.
  keca            - Compute the principal components of KECA method.
  kernel          - Compute a kernel given input data.
  kernelcentering - Centered kernels.
  kmnf            - Compute the principal components of KMNF method.
  kopls           - Compute the principal components of KOPLS method.
  kpca            - Compute the principal components of KPCA method.
  kpls            - Compute the principal components of KPLS method.
  mnf             - Compute the principal components of MNF method.
  noise           - Predits a noise estimation of a data set.
  opls            - Compute the principal components of OPLS method.
  pca             - Compute the principal components of PCA method.
  pinwheel        - Generate original data.
  pls             - Compute the principal components of PLS method.
  PredCCA         - Predicts labels by means of the projected data onto principal components of CCA method.
  PredKCCA        - Predicts labels by means of the projected data onto principal components of KCCA method.
  PredKECA        - Predicts labels by means of the projected data onto principal components of KECA method.
  PredKMNF        - Predicts labels by means of the projected data onto principal components of KMNF method.
  PredKOPLS       - Predicts labels by means of the projected data onto principal components of KOPLS method.
  PredKPCA        - Predicts labels by means of the projected data onto principal components of KPCA method.
  PredKPLS        - Predicts labels by means of the projected data onto principal components of KPLS method.
  PredMNF         - Predicts labels by means of the projected data onto principal components of MNF method.
  PredOPLS        - Predicts labels by means of the projected data onto principal components of OPLS method.
  PredPCA         - Predicts labels by means of the projected data onto principal components of PCA method.
  PredPLS         - Predicts labels by means of the projected data onto principal components of PLS method.
  simfeat         - Demo in order to comparing the richness of extraction with differents methods.

  
%%% REFERENCES %%%

 I. T. Jollife, Principal Component Analysis, Springer, 1986.

 J. Shawe-Taylor and N. Cristianini, Kernel Methods for Pattern Analysis, Cambridge University Press, 2004.

 C. M. Bishop, Pattern Recognition and Machine Learning (Information Science and Statistics), Springer-Verlag New York, Inc., 2006.

 O. Chapelle, B. Schölkopf, and A. Zien, Semi-Supervised Learning, MIT Press, Cambridge, 1st edition, 2006.

 G. Camps-Valls and L.Bruzzone, "Kernel Methods for Remote Sensing Data Analysis", John Wiley and Sons, Ltd. 2009.

 G. Camps-Valls, D. Tuia, L. Gómez-Chova, S. Jiménez, J. Malo, "Remote Sensing Image Processing. Synthesis Lectures on Image, Video, and Multimedia Processing",  Morgan & Claypool Publishers,2012.

 S. Wold, C. Albano, W. J. Dunn, U. Edlund, K. Esbensen, P. Geladi, S. Hellberg, E. Johansson, W. Lindberg, and M. Sjostrom, Chemometrics, Mathematics and Statistics in Chemistry, chapter Multivariate Data Analysis in Chemistry, p. 17, Reidel Publishing Company, 1984.

 R. Rosipal and N. Krämer, “Overview and recent advances in partial least squares,” in Subspace, Latent Structure and Feature Selection. 2006, vol. 3940 of LNCS, pp. 34–51, Springer.

 K. Worsley, J. Poline, K. Friston, and A. Evans, “Characterirzing the response of PET and fMRI data using multivariate linear models (mlm),” NeuroImage, vol. 6, pp. 305–319, 1998.

 A.A. Green, M. Berman, P. Switzer, and M.D. Craig. A transformation for ordering multispectral data in terms of image quality with implications for noise removal. IEEE Trans. Geosci. Rem. Sens., 26 (1): 65–74, Jan 1988b.
 
 M.A. Kramer. Nonlinear principal component analysis using autoassociative neural networks. AIChE Journal, 37 (2): 233–243, 1991.
  R. Rosipal and L.J. Trejo, “Kernel partial least squares regression in reproducing kernel Hilbert space,” J. Mach. Learn. Res., vol. 2, pp. 97–123, March 2002.
 
 J. Arenas-García and G. Camps-Valls,"Eficient Kernel Orthonormalized PLS for Remote Sensing Applications," IEEE Geoscience and Remote Sensing, vol. 46, no. 10, pp. 2872-2881, 2008.
 
 L. Gómez-Chova, A.A. Nielsen, and G. Camps-Valls. Explicit signal to noise ratio in reproducing kernel Hilbert spaces. In IEEE Geosc. Rem. Sens. Symp. (IGARSS), pages 3570–3570. IEEE, Jul 2011c.
 
 Robert Jenssen, "Kernel Entropy Component Analysis", IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, Vol. 32, Nº. 5, May 2010.
 
 Luis Gómez-Chova, Robert Jenssen, and Gustavo Camps-Valls, "Kernel Entropy Component Analysis in Remote Sensing Image Clustering" IEEE Geoscience and Remote Sensing Letters, 9(2), 312 - 316, 2012.

