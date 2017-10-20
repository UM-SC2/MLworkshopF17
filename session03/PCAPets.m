clear all; close all; clc;
load dogData.mat;
load catData.mat;

% each column of the data are the vectorized image of a cat/dog. Each data
% set has 80 pets.
%% Let's take a look at the cats and dogs

for j = 1:9
    figure(1)
    subplot(3,3,j);
    dg = reshape(dog(:,j),64,64); %reshape the matrix into 64 by 64 image
    imshow(dg);
    figure(2)
    subplot(3,3,j);
    ct = reshape(cat(:,j),64,64);
    imshow(ct);
end
%% Constructing Data Matrix X
%1. Construct data matrix X such that each rows are the vectorized image
% Stack both the cat and dogs data matrix together

X=double(X); %convert data to double type.

%2. Singular Value Decomposition of X


figure(3)
% 3. plot the normalized singular values of X and see correlation structure
% of our data matrix. This shows how much variance of the data are
% explained by each PCs.
xlabel('PCs')
ylabel('% Variance')

%% 
%The PCs are in decreasing order of variance explained in the data
figure(4)

% 4. Reconstruct the first 4 principal components into image
%hint: imagesc() and colormap hot
    
    
    
    



%% How similar are cats and dogs?
%5. Project the data on the first three PCs


%6. Plot the first three PCs projections (3D plot), color cats and dogs
%with different colors.

%7. Repeat this using other PCs, 
%Are the separation between the two groups better or worse?





