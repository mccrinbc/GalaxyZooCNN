close all;
clear all; 

%REQUIREMENTS: 
    % IMAGE PROCESSING TOOLBOX

PATH = "/Users/brianmccrindle/Documents/736/galaxy-zoo-the-galaxy-challenge";
cd(PATH);

info = readmatrix('training_solutions_rev1.csv');

for ii = 1:length(info)
    img = imread(strcat(PATH,'/images_training_rev1/',sprintf('%d',info(ii,1)),'.jpg'));
    
    img_g = rgb2gray(img); %convert the image into grayscale
    [row,col] = size(img_g);
    
    %inital mask
    mask = zeros(row,col);
    mask(100:350,100:350) = 1; %apply an arbitrary mask size "centered" at the middle pixel
    img_mask = img_g.*(im2uint8(mask)./255); %not sure why I need to do this but okay? 
    %imshow(img_mask)
    
   % figure()
    BW = imbinarize(img_mask); %finds a threshold that reduces the interclass variance b/w white and black
    %imshow(BW)
    
    %we apply a dilation to the blobs in the image to remove any inpurities
    %within the structure of the blob
    kernal = strel('disk',2,4); %using this for dilating inside galaxies
    img_dilate = imdilate(BW, kernal);
    %figure()
    %imshow(img_dilate)
    
    %looking for all of the connected componants, get centroids
    CC = bwconncomp(img_dilate); %using the image processing toolbox
    numBlobs = regionprops(CC,'centroid'); %stores in [COL,ROW]
    
    %Finding the Euclidean Distance between center of image and main blob
    %centroid, remove all other blobs
    distances = [];
    for j = 1:length(numBlobs)
        center = numBlobs(j).Centroid;
        x = center(1); y = center(2);
        euclid_dist = sqrt( (x-row/2)^2 + (y-col/2)^2 ); %row and col are the size of the image
        distances = [distances, euclid_dist];
    end
    
    %find the index of the important blob, and create a list of the index
    %location in numBlobs that it corresponds to
    impIndex = find(distances == min(distances)); %important blob
    blobNumber = [1:length(numBlobs)];
    blobNumber(impIndex) = [];
    
    %Removing those blobs!
    for k = blobNumber
        img_dilate(CC.PixelIdxList{k}) = 0;
    end
    
    %figure(4)
    filtered_img = img_mask.*(im2uint8(img_dilate)./255);
    %imshow(filtered_img)
    
    %figure(5)
    cropped_img = imcrop(filtered_img,[110,110,190,190]);
    %imshow(croped_img)
   
    saveloc = strcat(PATH,'/preprocessed/',sprintf('%d',info(ii,1)),'.jpg');
    imwrite(cropped_img,saveloc);
end

