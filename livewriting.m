%% Gathering the Numbers
I = imread('scn.png');
gray = rgb2gray(I);         %Gray Scale
mBW = mean(gray(:));        
sBW = std(double(gray(:)));
th = mBW - 1*sBW;
BW = gray < th;             % Binary Image
I = medfilt2(BW,[1 1]);
I = imerode(I,strel('square',10));
I = imdilate(I,strel('square',5));
show(I)
%%
digits = [];
a=prdataset([]);
rpro = regionprops(I,'BoundingBox','Area');
j = 1;
for i=1:size(rpro)
    if rpro(i).Area > 1e3
        ob = imcrop(I,(rpro(i).BoundingBox));
        ob = imresize(ob,[240 240]);
        digits(:,:,j) = ob;
        rectangle('Position',rpro(i).BoundingBox,'EdgeColor','yellow')
        j = j+1;
    end
end
%%
live_dataset = im2obj(digits);
[o,f] = size(live_dataset);
labels_digits = [];
for i=1:o
    show(live_dataset(i,:))
    i_dgt = input('digit');
    labels_digits = [labels_digits i_dgt];
end

live_dataset = setlabels(live_dataset,labels_digits');
