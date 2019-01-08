clc;clear;
fusion_para = 0.3;
infolder = 'input_img360';
outfolder = 'predi_salgan360';
mkdir(outfolder);

fprintf('Predicting Global Saliency Maps\n');
mkdir('predi_global_salmap');
cmd = ['python ', '03-predict.py ' infolder ' ' 'predi_global_salmap'];
status = system(cmd);
if status ~= 0
    fprintf('There is something wrong with SALGAN. Please check and run again.\n');
end

imlist=dir(infolder);
for n=3:length(imlist)
    fileName = imlist(n).name 
    im360= imread([infolder '\' fileName ]);  
    fprintf('Read image.\n');

    [iml0 imw0 c] = size(im360);
    im360 = imresize(im360, [1024 2048]);
    headmove_h = (0:10:80);
    headmove_v = (0:10:80);
    vfov = 90;
    iml = size(im360,1);
    imw = size(im360,2);

    fprintf('Making multiple cube projection...\n');
    mkdir('MCP');
    for hh = 1:length(headmove_h)
        offset=round(headmove_h(hh)/360*imw);
        im_turned = [im360(:,imw-offset+1:imw,:) im360(:,1:imw-offset,:)];
        for hv = 1:length(headmove_v)
            [out] = equi2cubic(im_turned, iml, vfov, headmove_v(hv));
            for i=1:6
                imwrite(cell2mat(out(i)), ['MCP\' num2str(hv) '_' num2str(hh) '_im360_', num2str(i), '.jpg']);
            end
        end      
    end
    clear out;

    % predict local salmap
    fprintf('Getting local salmap from SALGAN360 model...\n');
    mkdir('predictions_local_salmap');
    cmd = ['python ', '03-predict.py ' 'MCP'  ' predictions_local_salmap'];
    status = system(cmd);
    if status ~= 0
       fprintf('There is something werong with SALGAN. Please check and run again.\n');
    end

    fprintf('Fusing global and local salmaps...\n');
    saliencyList = dir('predictions_local_salmap');
    iml = size(imread([saliencyList(3).folder '\' saliencyList(3).name]),1);
    im_cub_sal = zeros(iml, iml, 6, 3);
    im_salgan_0 = zeros(iml,iml*2);
   
    for v = 1:length(headmove_v)
        for h=1:length(headmove_h)
            for i=1:6
                filename = saliencyList(i+(v-1)*54+(h-1)*6+2).name
                cubsal = double(imread(['predictions_local_salmap\', filename]));
                cubsal = imresize(cubsal, [iml iml]);
                im_cub_sal(:,:,i,1) = cubsal;
                im_cub_sal(:,:,i,2) = cubsal;
                im_cub_sal(:,:,i,3) = cubsal;
            end    
           [hv, rest] = strtok(filename,'_');
           hv = str2double(hv);
           hh = str2double(strtok(rest,'_'));
           im_salgan = cubic2equi(0,im_cub_sal(:,:,5,:), im_cub_sal(:,:,6,:), im_cub_sal(:,:,4,:), im_cub_sal(:,:,2,:), im_cub_sal(:,:,1,:), im_cub_sal(:,:,3,:));
           out = equi2cubic(im_salgan, iml, vfov, -headmove_v(hv));
           im_salgan = cubic2equi(-headmove_h(hh),cell2mat(out(5)),cell2mat(out(6)),cell2mat(out(4)),cell2mat(out(2)),cell2mat(out(1)),cell2mat(out(3)));
           im_salgan = im_salgan(:,:,1);
           im_salgan = double(im_salgan)+im_salgan_0;
           im_salgan_0 = im_salgan;
        end
    end
    im_salgan = im_salgan./(h*v);
    im_salgan = im_salgan/max(max(im_salgan));
    im_salgan = imresize(im_salgan, [1024 2048]);

    Gsalmap = double(imread(['predictions_global_salmap/' fileName]));
    Gsalmap = imresize(Gsalmap, [1024 2048]);
    Lsalmap = im_salgan;
    Lsalmap = Lsalmap./max(Lsalmap(:)).*255;
    Csalmap = fusion_para.*Gsalmap+(1-fusion_para).*Lsalmap;
    Csalmap = Csalmap./max(Csalmap(:))*255;
    imwrite(uint8(im_new),[outfolder '\' fileName]);    

    fprintf('Done!');
end
