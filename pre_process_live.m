function [processed_dataset, data] = pre_process_live(dataset, image_size)
  
    %% prnist
    a = dataset;    
    lab = {{}};
    for i = 1:size(a,1)
            
        img = data2im(a(i,:));
       % bouding box
       img = im_box(img,[10,10,10,0]);
            
       % Remove noise
       SE = strel('disk',1);
       img = imclose(img,SE);
       img = imerode(img,SE);
       img = imdilate(img,SE);

        % Slant correction
        m = im_moments(img,'central');
        var_x = m(1); 
        var_y = m(2); 
        cov_xy = m(3);
        theta = atan(2*cov_xy/(var_x - var_y));
        T = affine2d([1 0 0; sin(.5*pi-theta) cos(.5*pi-theta) 0; 0 0 1]);
        img = imwarp(img,T); 

        % Pre-process dataset (make all images square and bring them to same size)
        img = im_box(img,0,1);
        img = im_resize(img,[image_size, image_size]);
        img = im_box(img,1,0);

        data(i,:)=img(:);
        lab{i}= strcat('digit_',num2str(getlabels(a(i,:))));
  
    end    
    % Return processed data
    processed_dataset = prdataset(data,lab);
end