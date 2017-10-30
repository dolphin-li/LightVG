% This script demosntrates how the kernel can be optimized for
% Green's function.
% 
% REQUIRES: Image Processing Toolbox , Optimization Toolbox
% 
% In order to improve the results, change the number of iterations &
% initial guesses in optGreensKernel.m
% 
% For more information and the updated kernels for different 
% convolution problems, please visit the project website:
% 
% http://www.cs.huji.ac.il/labs/cglab/projects/convpyr
% 

clear all;
close all;

%% Load data
loadData;

%% Initial guess
    
% Starting with better initial guess (ie, gaussian kernels for h1/h2 for
% example can make the convergence faster, but typically random point is
% good enough to produce working kernels.
w0 = rand(1,5);    
    

%% Multi-scale optimization
for d = 4:11    
    dim = 2^d;
    
    if ( size(I,1) ~= dim)    
        cI = imresize(I, [dim dim]);
        cI = padarray(cI, [ps ps]);
        
        % Compute the divergence of the gradient field of I
        dx_f = imfilter(cI ,[1 -1 0]);
        dy_f = imfilter(cI ,[1 -1 0]');
        divG = imfilter(dx_f, [0 1 -1]) + imfilter(dy_f, [0 1 -1]'); 
    end

    % Optimization
    my_fun = @optHandle;
    w0 = fminunc(my_fun,double(w0));
    
end

%% Test image
[h1,h2,g] = constructKernels( w0 );

res = evalf( -divG1, h1, h2, g );    
figure; imshow(res); title('Reconstruction of the test image');
