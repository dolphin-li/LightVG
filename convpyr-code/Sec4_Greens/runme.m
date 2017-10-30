
clear all; close all;

%% Construct filter set: h1, h2 and g

% Choose computational budget

% This filter set was optimized under computational budget
% of the same 5x5 separable filter for up/down-sampling and 3x3 
% separable filter for solution on each band.
% This is the most compact filter set we have found so far that
% already produces high-quality reconstruction.
budget = 1; % Low

% This filter set was optimized under computational budget
% of the 7x7/5x5 separable symmetric filters.  
% budget = 2; % High

if budget == 1

    % These numbers define F_5
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    w = [0.15 0.5 0.7   0.175 0.547];    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    h1 = w(1:3);
    h1 = [h1 h1(end-1:-1:1)];
    h1 = h1' * h1;

    h2 = h1;

    g = w(4:5);
    g = [g g(end-1:-1:1)];
    g = g' * g;
else

    % These numbers define F_7
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    w = [0.06110 0.26177 0.53034 0.65934  ...
            0.51106 0.05407 0.24453 0.57410];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    h1 = w(1:4);
    h1 = [h1 h1(end-1:-1:1)];
    h1 = h1' * h1;

    h2 = h1*w(5);

    g = w(6:8);
    g = [g g(end-1:-1:1)];
    g = g' * g;

end

%% Load data

% Load file
% fname = 'horses.jpg';
fname = 'children.jpg';

im = double(imread(fname))/255;

for i = 1:3

    I = im(:,:,i); 
    
    % Simplest way to ensure compactness of the divergence.
    % Treatment of boundary conditions changes according
    % to this.
    ps = 1;
    I = padarray(I, [ps ps]);

    % Compute the divergence of the gradient field of I
    dx_f = imfilter(I ,[1 -1 0]);
    dy_f = imfilter(I ,[1 -1 0]');
    divG = imfilter(dx_f, [0 1 -1]) + imfilter(dy_f, [0 1 -1]'); 

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Transform ( - this is the core of our method )
    Ir = evalf( -divG, h1, h2, g );
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Set the mean of the original
    Ir = Ir-mean(Ir(:))+mean(I(:));
    
    res(:,:,i) = Ir;
    cdivG(:,:,i) = divG;

end

% Crop to the original size
res = res(ps+1:end-ps,ps+1:end-ps, :);
cdivG = cdivG(ps+1:end-ps,ps+1:end-ps, :);

% Show side-by-side
figure; imshow(im); title('Original Image');
figure; imshow(cdivG); title('Recontruction Input');
figure; imshow(res); title('Reconstruction Result');

