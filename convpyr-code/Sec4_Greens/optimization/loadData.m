global I;
global cI;
global divG;

global I1;
global divG1;


%%% Training Data

path = './';   
fname = 'box.jpg';

im = double(imread([path fname]))/255;

I = rgb2gray(im);

% Simplest way to ensure compactness of the divergence.
% Treatment of boundary conditions changes according
% to this.
ps = 1;
I = padarray(I, [ps ps]);

% Compute the divergence of the gradient field of I
dx_f = imfilter(I ,[1 -1 0]);
dy_f = imfilter(I ,[1 -1 0]');
divG = imfilter(dx_f, [0 1 -1]) + imfilter(dy_f, [0 1 -1]'); 

%%% Test Data
   
fname = 'chart.jpg';
im = double(imread([path fname]))/255;

I1 = rgb2gray(im);

% Simplest way to ensure compactness of the divergence.
% Treatment of boundary conditions changes according
% to this.
ps = 1;
I1 = padarray(I1, [ps ps]);

% Compute the divergence of the gradient field of I
dx_f = imfilter(I1 ,[1 -1 0]);
dy_f = imfilter(I1 ,[1 -1 0]');
divG1 = imfilter(dx_f, [0 1 -1]) + imfilter(dy_f, [0 1 -1]'); 