
% This script demonstrates high-dynamic range image compression
% by [Fattal et al. 2002] can be carried out by the means of
% Convolution Pyramids by [Farbman et al. 2011]
%   For more details: 
%   http://www.cs.huji.ac.il/labs/cglab/projects/convpyr

clear all

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


hdr = double(hdrread('./StillLife.hdr'));

L = 0.2989*hdr(:,:,1) + 0.587*hdr(:,:,2) + 0.114*hdr(:,:,3) + 10e-6;
I = log(L+10e-6); 

% Add padding
ps = 1;
I = padarray(I, [ps ps]);

% Tone-mapping parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is a bunch of knobs that tweak the tone-mapping process.
% These parameters are extra-sensitive and in order to get the
% results you like the most, you may need to play with them.
% For more details, see [Fattal et al. 2002]
alpha = 0.000001; % Bellow this gradient considered small
beta = 0.895;         % Compression parameter
a = 0.5;  % percentile to min mapping
b = 0.99; % max mapping
s = 0.45; % Saturation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute F which is used to attenuate the gadients
F = AttenuateGradients(I, alpha, beta);

% Compute the gradients, manipulate them
dx_f = imfilter(I ,[1 -1 0]);
dx_f = dx_f.*F;
dy_f = imfilter(I ,[1 -1 0]');
dy_f = dy_f.*F;

% Compute the divergence of the manipulated gradient field
divG = imfilter(dx_f, [0 1 -1]) + imfilter(dy_f, [0 1 -1]'); 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Transform ( - this is the core of our method )
res = evalf( -divG, h1, h2, g );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Remove padding
I = I(ps+1:end-ps,ps+1:end-ps);
res = real(res(ps+1:end-ps,ps+1:end-ps));
% You can probably live without getting the exponent back
res = exp(res); 

% Mapping the image to the proper range. It sets the brightness/contrast of
% the tone-mapped image.
tmp = res(:);
stmp = sort(tmp);
mn =  stmp( round(length(tmp)/100 * a));
mx =  max(tmp*b);
res = (res-mn)./(mx-mn);

% Restore the color & tweak the saturation
for i=1:3
    cres(:,:,i) = res .* abs((hdr(:,:,i)./L)).^s;
end

figure(1); imshow(cres);
% imwrite(cres, 'res.png');

