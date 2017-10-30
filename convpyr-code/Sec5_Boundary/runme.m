
% This script demonstrates how an effect similar to that of
% Seamless Image Cloning by [Perez et al. 2002] can be implemented
% as an instance of Shepard's method (using ratio of convolutions
% formulation). This ratio in turn can be efficiently approximated
% using Convolution Pyramids by [Farbman et al. 2011]
%   For more details: 
%   http://www.cs.huji.ac.il/labs/cglab/projects/convpyr

clear all

%% Load files

src = double(imread('source.jpg'))/255;
ftrg = double(imread('target.jpg'))/255;
mask = imread('mask.png'); mask = logical(mask(:,:,1));


%% Cut pixels from the source & paste to the target image

% Choose upper-left corner and cut a part of the image
sm = size(mask);
posx = 405; posy = 333;
trg = ftrg(posy:posy+sm(1)-1, posx:posx+sm(2)-1,:);

% Copy pixels
temp = trg;
for i=1:3
    
    sr = src(:,:,i);
    cr = temp(:,:,i);
    cr(mask) = sr(mask);
    temp(:,:,i) = cr;
    
end
res_copy = ftrg;
res_copy(posy:posy+sm(1)-1, posx:posx+sm(2)-1,:) = temp;

figure(1); imshow(res_copy); title('Cut&Paste Pixels');

%% Seamless Cloning - Poisson 
% or equivalently Laplace eq. with trg-src providing Dirichlet
% boundary conditions.

% Characteristic function: 1 on the boundary, 0 otherwise
h = fspecial('laplacian', 0);
chi = imfilter(double(mask),h);
chi(chi<0) = 0;
chi(chi>0) = 1;

% ldp debug
figure;
imshow(chi);
tmpchi = conv2(chi, ones(3, 1) / 3);
tmpchi = conv2(tmpchi, ones(1, 3) / 3);
figure;
imshow(tmpchi);

% Error
erf = trg - src;

res = zeros(size(erf));
for i=1:3
    
    sr = src(:,:,i);
    tr = trg(:,:,i);   
    
    a = erf(:,:,i);
    a(~chi) =  0;

    % Laplace eq. with Dirichlet bc 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Ierf = LaplacianDirichlet(a,mask);
    temp = Ierf + sr;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    tr(mask) = temp(mask);
    res(:,:,i) = tr;
end

res_rbf = ftrg;
res_rbf(posy:posy+sm(1)-1, posx:posx+sm(2)-1,:) = res;
figure(2); imshow(res_rbf); title('Poisson');
% imwrite(res_rbf, '_res_poisson.png');

%% Seamless Cloning - Shepard's method 

% Create Kernel. Should cover the mask
hh = zeros(sm);
hh(round(sm(1)/2), round(sm(2)/2)) = 1.0;
hh = double(bwdist(hh));

% Changing exponent or the bias can significantly affect the 
% results the exact form of the kernel is a matter of taste in 
% this problem.
hh = 1./((hh+0.1).^3); 

res = zeros(size(erf));
for i=1:3
    
    sr = src(:,:,i);
    tr = trg(:,:,i);   
    
    a = erf(:,:,i);
    a(~chi) =  0;

    % Shepard's interpolation via convolution 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Ierf = fftimfilter(a,hh);
    Ichi = fftimfilter(chi,hh);
    temp = Ierf./Ichi + sr;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    tr(mask) = temp(mask);
    res(:,:,i) = tr;
end

res_rbf = ftrg;
res_rbf(posy:posy+sm(1)-1, posx:posx+sm(2)-1,:) = res;
figure(3); imshow(res_rbf); title('Shepard method');
% imwrite(res_rbf, '_res_rbf.png');

%% Seamless Cloning - Convolution Pyramid

% Filter set: h1, h2 and g
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w = [0.1507 0.6836 1.0334 0.0270 0.0312 0.7753];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

h1 = w(1:3);
h1 = [h1 h1(end-1:-1:1)];
h1 = h1' * h1;

h2 = h1*w(4);

g = w(5:end);
g = [g g(end-1:-1:1)];
g = g' * g;

for i=1:3
    
    sr = src(:,:,i);
    tr = trg(:,:,i);   
    
    a = erf(:,:,i);
    a(~chi) =  0;
    
    % Convolution Pyramid 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Ierf = evalf( a, h1, h2, g );
    Ichi = evalf( chi, h1, h2, g );
    temp = Ierf./Ichi + sr;    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    tr(mask) = temp(mask);
    res(:,:,i) = tr;
end

res_fc = ftrg;
res_fc(posy:posy+sm(1)-1, posx:posx+sm(2)-1,:) = res;
figure(4); imshow(res_fc); title('Fast Convolution');
% imwrite(res_fc, '_res_cp.png');
