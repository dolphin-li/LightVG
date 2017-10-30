% This script demosntrates how the kernel can be optimized for
% a specific value of sigma on a 128x128 grid. 
% 
% REQUIRES: Image Processing Toolbox , Optimization Toolbox
% 
% In order to improve the results, change the number of iterations &
% initial guesses in optGaussKernel.m
% 
% In the ./kernels directory you can find kernels pre-optimized for 
% different values of sigma.
% 
% For the details on why gaussian case needs a special treatment
% in our framework and for the reason that we need to introduce a
% per-level scaling factors please take a look at the paper (Section 6).
% 
% For more information and the updated kernels for different 
% convolution problems, please visit the project website:
% 
% http://www.cs.huji.ac.il/labs/cglab/projects/convpyr
% 

clear all;
close all;

%% Optimize

sigma = 5;
w = optGaussKernel( sigma );

%% Construct a signal - 3 spikes 

n = 128; % Should match the optimization resolution
in = zeros(n,n);
in(n/2, n/2) = 1;   
in(n/2, n/2+n/4) = 1;  
in(n/2, n/2-n/4) = 1;   

%% Compute the exact convolution

[X,Y] = meshgrid(-n:n, -n:n);  
h = exp(-(X.^2 + Y.^2)/(2*sigma^2));
EC = imfilter(in, h,0); 

%% Compute the approximate convolution

AC = evalg( in, w );    

%% Show the reslut

figure('name','Convolution with Gaussian Kernel','numbertitle','off')
subplot(2,2,3); imagesc(EC); title('Exact Convolution'); 
subplot(2,2,4); imagesc(AC); title('Approx. Convolution');
subplot(2,2,1); imagesc(in); title('Input Signal');

subplot(2,2,2); hold on;
title('1D scan'); legend('Exact','Approx.');
plot(EC(end/2, :), 'r');
plot(AC(end/2, :), 'g');
