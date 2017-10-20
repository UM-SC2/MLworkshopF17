clear all; close all; clc;
load dogData.mat;
load catData.mat;

%% Let's take a look at the cats and dogs

for j = 1:9
    figure(1)
    subplot(3,3,j);
    dg = reshape(dog(:,j),64,64);
    imshow(dg);
    figure(2)
    subplot(3,3,j);
    ct = reshape(cat(:,j),64,64);
    imshow(ct);
end
%% Constructing Data Matrix X
X=[cat'; dog'];
X=double(X);
[u,s,v]=svd(X, 'econ');

% plot the normalized singular values of X and see correlation structure
% of our data matrix.
figure(3)
plot(diag(s)*100/sum(diag(s)),'ko','Linewidth',2)
xlabel('Singular Values')
ylabel('% Variance')

%% 
%The PCs are in decreasing order of variance explained in the data
figure(4)
for j=1:4
    subplot(2,2,j) 
    imagesc(reshape(v(:,j),64,64))
    
    colormap hot
    title(sprintf('PC %d',j))
    
end


%% How similar are cats and dogs?
%First project the data on the first three PCs
T = X*v(:,1:3)

%
figure;
plot3(T(1:80,1),T(1:80,2),T(1:80,3),'ro'); hold on;
plot3(T(81:160,1),T(81:160,2),T(81:160,3),'ko');
xlabel('PC1');
ylabel('PC2');
zlabel('PC3');
%Repeat this using other PCs, 
%Are the separation between the two groups better or worse?





