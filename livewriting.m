%% Automatically cut numbers and dd labels
clear all; close all; clc;
% Load image and convert to black and white image
I = imread('img/scn.png');  % Load image
J = rgb2gray(I);        % convert to gray Scale
m = mean(J(:));        
s = std(double(J(:)));
I = J < (m - 1*s);
% Filter some noise
I = medfilt2(I,[1 1]);
I = imerode(I,strel('square',10));
I = imdilate(I,strel('square',5));
show(I) % show the scan
% set bouding box
nmbr=[];
rp = regionprops(I,'BoundingBox','Area');
j = 1;
test = imcrop(I,(rp(10).BoundingBox));
imshow(test);
% Apply bounding box
for i=1:size(rp)
    if rp(i).Area > 1e3
        ob = I;
        ob = imcrop(ob,(rp(i).BoundingBox));
        % Resizing  
        [H,W] = size(ob);
        tmp = zeros(max(H,W),max(H,W), 'logical');
        
        [HH, WW] = size(tmp);
        hdiff = (HH - H) / 2;
        wdiff = (WW - W) / 2;
        
        %% Center digit
        if hdiff > 0
            if floor(hdiff) == hdiff
                startH = 1 + hdiff;
                endH = H + hdiff;
            else
                startH = ceil(hdiff);
                endH = H + floor(hdiff);
            end
        else
            startH = 1;
            endH = H;
        end
        
        if wdiff > 0
            if floor(wdiff) == wdiff
                startW = 1 + wdiff;
                endW = W + wdiff;
            else
                startW = ceil(wdiff);
                endW = W + floor(wdiff);
            end
        else
            startW = 1;
            endW = W;
        end
        
        tmp(startH:endH,startW:endW)=ob;
        ob = tmp;
        ob = imresize(ob,[240 240]);
        nmbr(:,:,j) = ob;
        rectangle('Position',rp(i).BoundingBox,'EdgeColor','yellow')
        j = j+1;
    end
end

%% Add labels to the digits
img = im2obj(nmbr);
lab = [];
for i=1:size(img)
    show(img(i,:))
    i_nmbr = input('digit_');
    lab = [lab i_nmbr];
end
live_dataset = setlabels(img,lab');