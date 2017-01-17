%Robust multi-model homography estimation and virtual insertion of planar objects

%Using VLFeat library, http://www.vlfeat.org/
%See sample results at http://www.youtube.com/watch?v=YmCKqXfdJ8w&feature=youtube_gdata

%This script replaces instances of im2 found within im1 with imNew, to various degrees of accuracy

%---------------------------------------------------------------------
%Flags, options
THRESHOLD = 5.4; %Higher threshold is less strict
MINDIST = 20000;
DETECT_INLIERS=1; %Set to 1 to use manual rejection scheme, from hw2
ITERATIONS = 2; %How many times to run it on each 
UBCTHRESH = 0; %Threshold of UBC_MATCH; default is 1.5

%Image inputs
 im1 = imread(fullfile(vl_root, 'data', 'cs174bhw3', '003.png')) ;
 im2 = imread(fullfile(vl_root, 'data', 'cs174bhw3', 'yellow.png')) ;
 imNew = imread(fullfile(vl_root, 'data', 'cs174bhw3', 'blue1.png')) ;
 
 %--------------------------------------------------------------------
 
% make single
im1_ = im2single(im1) ;
im2_ = im2single(im2) ;
% make grayscale
if size(im1_,3) > 1, im1g = rgb2gray(im1_) ; else im1g = im1_ ; end
if size(im2_,3) > 1, im2g = rgb2gray(im2_) ; else im2g = im2_ ; end

% Sift and find correspondences (Project 2 code is here)
% Sift is a function from VL_Feat that returns the SURF descriptors found in image
[f1,d1] = vl_sift(im1g) ;
[f2,d2] = vl_sift(im2g) ;

 if size(im1_,3) > 1, im1g = rgb2gray(im1_) ; else im1g = im1_ ; end
 im1_ = im2single(im1) ;

 
[matches, scores] = vl_ubcmatch(d1,d2, UBCTHRESH);
%vl_sift returns descriptors, which we in turn put into the vl_ubcmatch function. vl_ubmatch
%takes in two descriptors, and an optional threshold, which determines the how strict it is in finding
%inliers. When this threshold is set to zero, there is no outlier rejection/inlier detection, and the matching
%screen generally displays a mess of correspondances, most which are clearly bad matches


%In order to sift out the outliers, and keep the inliers, we took advantage of the fact that the
%ubcmatch function also returns a 'scores' integer, which is the euclidian squared distance. Our outlier
%rejection scheme was a pair of simple loops. First, we start with an arbitrary distance, called
%min_distance which we narrow down to the smallest score, with a loop that goes through the scores.
%Then, compare each distance for each match with this benchmark, multiplied by an abritrary threshold
%that we set, similar to ubcmatch. We keep a match only if the distance is smaller than this product. For
%the river pictures that came with the vl_feat package, I chose a threshold of 6, which cut down my
%result significantly.

if(DETECT_INLIERS == 1),
    prematches = matches;
    numMatches = size(prematches,2);

    max_dist = 0; %Not used this time
    min_dist = MINDIST; 

    for i = 1:numMatches
    dist = scores(i);
    if dist<min_dist, min_dist = dist; end
    if dist>max_dist, max_dist = dist; end
    end

matches = [];

    for i=1:numMatches
     if scores(i)<THRESHOLD*min_dist, 
        matches = [matches, prematches(:,i)]; 
     end
    end
end

numMatches = size(matches, 2);

X1 = f1(1:2,matches(1,:)) ; X1(3,:) = 1 ;
X2 = f2(1:2,matches(2,:)) ; X2(3,:) = 1 ;

%Find Homography using RANdom SAmple Consensus (RANSAC)
%
clear H score ok ;
for t = 1:1000
  % estimate homography
  subset = vl_colsubset(1:numMatches, 4) ;
  A = [] ;
  for i = subset
	xip = X2( 1, i );
    yip = X2( 2, i );
    wip = X2( 3, i );

    xi = X1( :, i );

    Ai = [ 0, 0, 0,    -wip * xi',   yip * xi' ;
           wip * xi',     0, 0, 0,  -xip * xi' ];

    A = [ A ; Ai ];

  end
  %Solve Ax=0 using singular value decomposition
  [U,S,V] = svd(A) ;
  %Take the last column of V, which should be the smallest 9 singular values, and reshape into 
  %a 3x3 matrix
  H{t} = reshape( V(:,9), 3, 3 )';

  % Score homography
  %This is the juice of the RANSAC method; Having taken four random correspondences (the minimum to form a homography) in this iteration, 
  %we have fitted a homography. Now, using this homography, we apply it to our image, and look for inlying points
  %Every iteration, we count the number of inliers, and select the model with the most inliers after 1-10k iterations as our homography
  X2_ = H{t} * X1 ;
  du = X2_(1,:)./X2_(3,:) - X2(1,:)./X2(3,:) ;
  dv = X2_(2,:)./X2_(3,:) - X2(2,:)./X2(3,:) ;
  ok{t} = (du.*du + dv.*dv) < 36;
  score(t) = sum(ok{t}) ;
  
currentH = (H{t});

  
  %SANITY CHECKS ON H MATRIX
  %In some of the trickiers matches, with outliers, like the checkerboard, we should throw out
  %obviously bad matches, like homography matrices that skew our image badly, or blow up our image
  skewRatio = currentH(1,1)/currentH(2,2);
if (skewRatio < 3/4 | skewRatio > 4/3 ),
 score(t) = 0;
end
 
if (abs(currentH(1,1))<2*abs(currentH(3,3)) | abs(currentH(2,2))<2*abs(currentH(3,3))),
 score(t) = 0;
end

D = diag(currentH);
if (sign(D(1)) ~= sign(D(2)) | sign(D(1))~=sign(D(2)) | sign(D(2))~=sign(D(3)))
 score(t) = 0;
end
 
end %End RANSAC iteration

[score, best] = max(score) 
H = H{best}
ok = ok{best} ;
diag(H)
dh1 = max(size(im2,1)-size(im1,1),0) ;
dh2 = max(size(im1,1)-size(im2,1),0) ;

figure(1) ; clf ;
%subplot(2,1,1) ;
imagesc([padarray(im1,dh1,'post') padarray(im2,dh2,'post')]) ;
o = size(im1,2) ;
line([f1(1,matches(1,:));f2(1,matches(2,:))+o], ...
     [f1(2,matches(1,:));f2(2,matches(2,:))]) ;
title(sprintf('%d tentative matches', numMatches)) ;
axis image off ;


% --------------------------------------------------------------------
% Mosaic
% This is taken from our previous mosaicing code; Having found the optimal homography matrix
% relating the two images, we apply it to imNew, and paste the new image over the correct match.
% --------------------------------------------------------------------
if (score>1),
    im2=imNew;
[a1 b1 c1] = size(im1);
[a2 b2 c2] = size(im2);

%TRANSFORM
%for im1
K = [1 b2 b2 1 ; 1 1 a2 a2; 1 1 1 1] ;
K_ = H\K;
K_(1,:) = K_(1,:) ./ K_(3,:) ; %Divide the 3rd row 
K_(2,:) = K_(2,:) ./ K_(3,:) ;
[u,v] = meshgrid(min([1 K_(1,:)]) : max([size(im1,2) K_(1,:)]),min([1 K_(2,:)]) : max([size(im1,1) K_(2,:)])) ;
%for im2
u_ = ((H(1,1) * u + H(1,2) * v + H(1,3))) ./ (H(3,1) * u + H(3,2) * v + H(3,3)) ; 
v_ = ((H(2,1) * u + H(2,2) * v + H(2,3))) ./ (H(3,1) * u + H(3,2) * v + H(3,3)) ; 

%imwbackward is a useful vlfeat function that returns the value of the image at the given point
im1_new = vl_imwbackward(im2double(im1),u,v) ; %vlfeat functions
im2_new = vl_imwbackward(im2double(im2),u_,v_) ;

%Check for any elements that are not numbers
%Isnan returns the locations of where, if any, elements are not a number
mass = ~isnan(im1_new) + ~isnan(im2_new) ;
im1_new(isnan(im1_new)) = 0 ; %otherwise, we get large blank patches
im2_new(isnan(im2_new)) = 0 ;
mosaic = (im1_new + im2_new)./mass;


figure(2) ; clf ;
imagesc(mosaic) ; axis image off ;
title('Mosaic') ;
end;
if (score<=1)
    error = 'No good matches'
end
%im1 = mosaic;
% end
