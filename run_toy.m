clear all
close all
% clc

addpath('./code_coregspectral');
projev = 1; 

%% UCI
dataset='BBCSport.mat';
numClust=2;
num_views=2;
load(dataset);

X1=data{1};
X2=data{2};
truth=truelabel{1}';

sigma(1)=optSigma(X1);
sigma(2)=optSigma(X2);
% sigma(1)=10
% sigma(2)=10

%% Construct kernel and transition matrix
K=[];
T=[];
for j=1:num_views
    options.KernelType = 'Gaussian';
    options.t = sigma(j);
    K(:,:,j) = constructKernel(data{j}',data{j}',options);
    D=diag(sum(K(:,:,j),2));
    L_rw=D^-1*K(:,:,j);
    T(:,:,j)=L_rw;
end

%% SSC with pairwise co-regularization
fprintf('======================================\n');
beta = 0.1;
alpha = 0.1;
fprintf('Running with PSSC, alpha = %f, beta = %f.\n',alpha,beta);
[F P R nmi avgent AR,Q] = ssc_p(K,num_views,numClust,alpha,beta,truth,projev);
% plotmatrix(Q(:,:,1),4)
title('ssc-p')


% 
% 
% %% co-regspectral multiview spectral
% numiter = 10 ;
% fprintf('======================================\n');
% fprintf('Co-regspectral multiview spectral\n');
% co_sigma=[100 100];
% co_sigma = sigma;
% lambda=1;
% lambda = 0.01
% [F P R nmi avgent AR,Q] = spectral_pairwise_multview(data,num_views,numClust,co_sigma,lambda,truth,numiter);


