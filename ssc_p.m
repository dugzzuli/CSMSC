function [F P R nmi avgent AR,Q] = ssc_p(K,num_views,numClust,alpha,beta,truth,projev)
% INPUT:
% OUTPUT:    
    numEV = numClust;
    numVects = numClust;      

    N = size(K,1);
    I = eye(N);
    L = zeros(N,N,num_views);
    for i = 1 : num_views
        D = diag(sum(K(:,:,i),1));
        L(:,:,i) = I-sqrt(inv(D))*K(:,:,i)*sqrt(inv(D));
%         plotmatrix(K(:,:,i),i)
    end
%     pause
    opts.disp = 0;
    [Q] = solve_sscp2(L,num_views,alpha,beta,numClust);  
    save Q Q
%     load Q
%     Q(:,:,1) = (abs(Q(:,:,1))+abs(Q(:,:,1)'))/2;
    
%     [V Eval F P R nmi avgent AR C] = baseline_spectral_onkernel(Q(:,:,1),numClust,truth);
%     QQ = zeros(N,N);
%     for i = 1 :num_views
%         QQ = QQ + abs(Q(:,:,i));
%     end
%     [V Eval F P R nmi avgent AR C] = baseline_spectral_onkernel(QQ,numClust,truth);
%     [V Eval F P R nmi avgent AR C] = baseline_spectral_onRW(Q(:,:,1),numClust,truth,projev);
%     fprintf('F=%f, P=%f, R=%f, nmi score=%f, avgent=%f,  AR=%f,\n',F(1),P(1),R(1),nmi(1),avgent(1),AR(1));

    
    
% %     save QV Q V
% %     load QV
   
numEV = numClust*projev;
 Vnorm = zeros(N,numClust,num_views);
    kmeans_avg_iter = 20;   
%     rho = 0.5;
    for j = 1 : num_views %   use the first view
        Qj = (Q(:,:,j)+Q(:,:,j)')/2;
%         Qj = BuildAdjacency(thrC(Qj,rho));
        [V, E] = eigs(Qj,numEV,'LA',opts);
        U = V(:,1:numClust);
%         U = V(:,1:numClust,j);
        norm_mat = repmat(sqrt(sum(U.*U,2)),1,size(U,2));
        %%avoid divide by zero
        for i=1:size(norm_mat,1)
            if (norm_mat(i,1)==0)
                norm_mat(i,:) = 1;
            end
        end
        U = U./norm_mat;
        Vnorm(:,:,j) = U;
        for i=1:20
            C = kmeans(U,numClust,'EmptyAction','drop');
            [A nmii(i) avgenti(i)] = compute_nmi(truth,C);
%             accu(i) = 1-Misclassification(truth,C);
            [Fi(i),Pi(i),Ri(i)] = compute_f(truth,C);
            [ARi(i),RIi(i),MIi(i),HIi(i)]=RandIndex(truth,C);
        end
        F = mean(Fi);
        P = mean(Pi);
        R = mean(Ri);
        nmi = mean(nmii);
        avgent = mean(avgenti);
        AR = mean(ARi);
        fprintf('view %d\n',j);
        fprintf('F: %f(%f)\n', F, std(Fi));
        fprintf('P: %f(%f)\n', P, std(Pi));
        fprintf('R: %f(%f)\n', R, std(Ri));
        fprintf('nmi: %f(%f)\n', nmi, std(nmii));
        fprintf('avgent: %f(%f)\n', avgent, std(avgenti));
        fprintf('AR: %f(%f)\n', AR, std(ARi));
    end        
          
    if (1)
        %%%%averaging of U1 and U2
        V = sum(Vnorm,3)/num_views;
        normvect = sqrt(diag(V*V'));
        normvect(find(normvect==0.0)) = 1;
        V = inv(diag(normvect)) * V;
        %U = U./repmat(sqrt(sum(U.*U,2)),1,numClust*2); % normalize
        for j=1:kmeans_avg_iter
            C = kmeans(V(:,1:numVects),numClust,'EmptyAction','drop');
            [Fj(j),Pj(j),Rj(j)] = compute_f(truth,C);
            CAi_j(j) = 1-compute_CE(C, truth); % clustering accuracy
            [Aj nmi_j(j) avgent_j(j)] = compute_nmi(truth,C);
            [RIj(j),ARj(j),MIj(j),HIj(j)]=RandIndex(truth+1,C);
        end
        F = mean(Fj);
        P = mean(Pj);
        R = mean(Rj);
        CAi = mean(CAi_j);
        nmi = mean(nmi_j);
        avgent = mean(avgent_j);
        AR = mean(ARj);
        fprintf('view %d\n',j);
        fprintf('F: %f(%f)\n', F, std(Fj));
        fprintf('P: %f(%f)\n', P, std(Pj));
        fprintf('R: %f(%f)\n', R, std(Rj));
        fprintf('CAi: %f(%f)\n', CAi, std(CAi_j));
        fprintf('nmi: %f(%f)\n', nmi, std(nmi_j));
        fprintf('avgent: %f(%f)\n', avgent, std(avgent_j));
        fprintf('AR: %f(%f)\n', AR, std(ARj));        
    end
    
    
    
    
    