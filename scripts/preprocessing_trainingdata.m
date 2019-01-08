clc,clear;
folder = 'C:\Users\fchang\Desktop\ICME18_SaliencyChallenge\Images';
outfolder = 'C:\Users\fchang\saliency360_data_45';
mkdir([outfolder '\image256x192']);
mkdir([outfolder '\salmap256x192']);
mkdir([outfolder '\fixmap256x192']);
imlist = dir([folder '\Stimuli' ]);

output_size = 256;
vfov = 90;
headmove_h = 0:45:45;
headmove_v = 0:45:45;
sal_wid = 2048;
sal_len = 1024;

for i = 3:length(imlist)
    im_filename = imlist(i).name;
    filenum = regexp(im_filename, '_', 'split');
    filenum = regexp(cell2mat(filenum(1)), 'P', 'split');
    filenum = cell2mat(filenum(2));
    
    if sum(str2double(filenum)) > 74
        state = 'val';
    else
        state = 'train';
    end
     
    % read image
    img = imread([folder '\Stimuli\' im_filename]);
    imw = size(img, 2);
    iml = size(img, 1);
    
    % read salmap
    fileId = fopen([folder '\HE\SalMaps\HEsalmap_' filenum '_2048x1024_32b.bin'], 'rb');
    buf = fread(fileId, sal_len * sal_wid, 'single');
    salmap = reshape(buf, [sal_wid, sal_len])';
    salmap = salmap./max(salmap(:));
    fclose(fileId);
    
    % read fixations
    fixmap = zeros(sal_len, sal_wid);
    M = csvread([folder '\HE\Scanpaths\L\HEscanpath_' filenum '.txt'], 1, 0);
    flen = size(M,1);
    for k = 1:flen-1
        fixpos = round(M(k, 2:3).* [sal_wid sal_len])+[1 1];
        fixmap(fixpos(2), fixpos(1))=1;  
    end
    M = csvread([folder '\HE\Scanpaths\R\HEscanpath_' filenum '.txt'], 1, 0);
    flen = size(M,1);
    for k = 1:flen-1
        fixpos = round(M(k, 2:3).* [sal_wid sal_len])+[1 1];
        fixmap(fixpos(2), fixpos(1))=1; 
    end
    
    for hh = 1:length(headmove_h)
        offset = round(headmove_h(hh)/360*imw);
        sal_offset = round(headmove_h(hh)/360*sal_wid);
        im_turned = [img(:, imw-offset+1:imw, :) img(:, 1:imw-offset, :)];
        sal_turned = [salmap(:, sal_wid-sal_offset+1:sal_wid) salmap(:, 1:sal_wid-sal_offset)];
        fix_turned = [fixmap(:, sal_wid-sal_offset+1:sal_wid, :) fixmap(:, 1:sal_wid-sal_offset, :)];
        for hv = 1:length(headmove_v)
            [out] = equi2cubic(im_turned, iml, vfov, headmove_v(hv));
            [sal_out] = equi2cubic(sal_turned, sal_len, vfov, headmove_v(hv));
            [fix_out] = equi2cubic(fix_turned, sal_len, vfov, headmove_v(hv));
            for f=1:6
                imwrite(imresize(cell2mat(out(f)), [192 256]), [outfolder '\image256x192\' state '_P' filenum '_' num2str(hv) num2str(hh) num2str(f) '.jpg']);
                imwrite(imresize(cell2mat(sal_out(f)), [192 256]), [outfolder '\salmap256x192\' state '_P' filenum '_' num2str(hv) num2str(hh) num2str(f) '.png']);
                f_out = cell2mat(fix_out(f));
                f_out(~isfinite(f_out))=0;
                f_out=logical(f_out);
                s = regionprops(f_out, 'Centroid');
                f_out_new = zeros(size(f_out));
                for j=1:length(s)
                    pos = round(s(j).Centroid);
                    f_out_new(pos(2), pos(1))=255;
                end
                imwrite(imresize(f_out_new, [192 256]), [outfolder '\fixmap256x192\' state '_P' filenum '_' num2str(hv) num2str(hh) num2str(f) '.png']);
            end
       end     
    end
end