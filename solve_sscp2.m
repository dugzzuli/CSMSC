function [P U]= solve_sscp2(L,m,alpha,beta,k,display)
% pairwise sparse spectral clustering, multi-view

% m - number of views
% k - number of clusters
% L - n*n*m Laplacian matrices of m views

if nargin < 6
    display = 0;
end
n = size(L,1);
tol = 1e-4; 
maxIter = 1000; 
mu = 1e-6;
max_mu = 1e10;
rho = 1.1;
P = zeros(n,n,m);
Q = P;
Y = P;
U = P;
iter = 0;
if display
    obj = zeros(maxIter,1);
end
while iter < maxIter
    iter = iter + 1;    
    % update P.
    sumQ = alpha*sum(Q,3);
    YL = Y+L;
    for i = 1 : m
        P(:,:,i) = solve_l1((sumQ+(mu-alpha)*Q(:,:,i)-YL(:,:,i))/mu,beta/mu);
    end
    
    % update Q.
    sumP = alpha*sum(P,3);
    for i = 1 : m
        temp = (sumP+(mu-alpha)*P(:,:,i)+Y(:,:,i))/mu;
        temp = (temp+temp')/2;
        [U(:,:,i),D] = eig(temp);
        Dr = cappedsimplexprojection(diag(D),k);
        Q(:,:,i) = U(:,:,i)*diag(Dr)*U(:,:,i)';
    end    
    leq = P-Q;
    stopC = max(max(max(abs(leq))));
    if display
%         obj(iter) = trace(P'*L)+alpha*sum(sum(sum(abs(P))));
        obj(iter) = 0;
        disp(['iter ' num2str(iter) ',obj=' num2str(obj(iter),'%2.1e') ...
            ',mu=' num2str(mu) ',stopC=' num2str(stopC,'%2.3e')]);
    end
    if stopC < tol 
        break;
        
    else
        Y = Y + mu*leq;
        mu = min(max_mu,mu*rho);
    end
end

if display
    obj(iter:end)=[];
    figure(1)
    plot(obj)
end

function x = solve_l1(u,w)
x = max(0,u-w) + min(0,u+w) ;