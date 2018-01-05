delete(gcp)
matlabpool 12
clc;
clear;
metricsFolder = 'saliency/code_forMetrics'
addpath(genpath(metricsFolder))

frame_cut=30;
dsname='diem';

save_base_dir = fullfile('/data/sunnycia/saliency_on_videoset/Train/metric-matlab', dsname);
mkdir(save_base_dir);
if strcmp(dsname,'videoset')==1
    base_dir='/data/sunnycia/saliency_on_videoset/Train/metric-matlab/videoset'; % for the usage of metric_statistics

    base_sal_dir = '/data/sunnycia/SaliencyDataset/Video/VideoSet/Results/saliency_map';
    dens_dir = strcat('/data/sunnycia/SaliencyDataset/Video/VideoSet/ImageSet/Seperate/density_fc/density-6');
    fixa_dir = strcat('/data/sunnycia/SaliencyDataset/Video/VideoSet/ImageSet/Seperate/fixation');
    all_in_one_fixation_directory = '/data/sunnycia/SaliencyDataset/Video/VideoSet/ImageSet/All_in_one/fixation'; % for computing sauc metric
end
if strcmp(dsname,'ledov')==1
    base_dir='/data/sunnycia/saliency_on_videoset/Train/metric-matlab/ledov'; % for the usage of metric_statistics
    
end

if strcmp(dsname,'ledov')==1
    base_dir='/data/sunnycia/saliency_on_videoset/Train/metric-matlab/ledov'; % for the usage of metric_statistics
    
end

if strcmp(dsname, 'diem')==1
    base_dir='/data/sunnycia/saliency_on_videoset/Train/metric-matlab/diem'; % for the usage of metric_statistics

    base_sal_dir = '/data/sunnycia/SaliencyDataset/Video/DIEM/saliency_map';
    dens_dir = strcat('/data/sunnycia/SaliencyDataset/Video/DIEM/density/image/sigma32');
    fixa_dir = strcat('/data/sunnycia/SaliencyDataset/Video/DIEM/fixation_map/image');
    all_in_one_fixation_directory = '/data/sunnycia/SaliencyDataset/Video/DIEM/fixation_map/All_in_one'; % for computing sauc metric
end
if strcmp(dsname, 'gazecom')==1
    base_dir='/data/sunnycia/saliency_on_videoset/Train/metric-matlab/gazecom'; % for the usage of metric_statistics

    base_sal_dir = '/data/sunnycia/SaliencyDataset/Video/GAZECOM/saliency_map';
    dens_dir = strcat('/data/sunnycia/SaliencyDataset/Video/GAZECOM/density/sigma32');
    fixa_dir = strcat('/data/sunnycia/SaliencyDataset/Video/GAZECOM/fixations');
    all_in_one_fixation_directory = '/data/sunnycia/SaliencyDataset/Video/GAZECOM/All_in_one/fixations'; % for computing sauc metric
end

% model_list = {'DENSITY';'SAM';'FANG8';'XU';'SALICON';'ITKO';'GBVS';'PQFT';'SUN';'ISEEL';'MDB';};
% model_list = {'DENSITY';'SAM';'XU';'SALICON';'ITKO';'GBVS';'PQFT';'SUN';'ISEEL';'MDB';'FANG2';};
% model_list = {'MOTION';'MSFUSION';'UNCERTAINTY';'SAM';'XU';'SALICON';'ITKO';'GBVS';'PQFT';'SUN';'ISEEL';'MDB';'FANG2';'DENSITY';};
% model_list = {'MOTION';'MSFUSION';'UNCERTAINTY';};
% model_list = {'v1';'v3';};
% model_list = {'FANG2';'DENSITY';'xu_dupext40';'SAM';'train_kldloss-kld_weight-100-batch-1_1510102029_usesnapshot_1509584263_snapshot-_iter_100000';'MOTION';'MSFUSION';'UNCERTAINTY';'SAM';'XU';'SALICON';'ITKO';'GBVS';'PQFT';'SUN';'ISEEL';'MDB';};
% model_list = {'vo-v3-2_train_kldloss_withouteuc-batch-1_1513084718_snapshot-_iter_150000'};
% model_list = {'vo-v4-2-base_lr-0.01-snapshot-20000-display-1-batch-8_1513218849_snapshot-3000'};
% model_list = {'vo-v4-2-base_lr-0.01-snapshot-20000-display-1-batch-8_1513218849_snapshot-13500_threshold0.75'};
% model_list = {'vo-v4-2-base_lr-0.01-snapshot-20000-display-1-batch-8_1513218849_snapshot-13500_threshold0.75_rangesmooth'}
% model_list = {'vo-v4-2-snapshot-999999-display-1-ledovSet-batch-8_1513764025_snapshot-11250_threshold0'};
% model_list = {'vo-v4-2-resnet-snapshot-2000-display-1--batch-2_1514034705_snapshot-_iter_72000_threshold0';
% 'vo-v4-2-snapshot-2000-display-1--batch-8_1514033989_snapshot-_iter_20000_threshold0';};
% model_list = {'vo-v4-2-snapshot-2000-display-1-fulldens-batch-8_1514129167_snapshot-_iter_28000_threshold0'}
% model_list = {'vo-v4-2-resnet-catfeat-snapshot-2000-display-1--batch-2_1514034491_snapshot-_iter_72000_threshold0'}
% model_list = {'vo-v4-2-resnet-base_lr-0.01-snapshot-2000-display-1--batch-2_1514260519_usesnapshot_1514034705_snapshot-_iter_72000_snapshot-_iter_96000_threshold0'}
% model_list = {  'vo-v4-2-resnet-base_lr-0.01-snapshot-2000-display-1--batch-2_1514260519_usesnapshot_1514034705_snapshot-_iter_72000_snapshot-_iter_454000_threshold0';
%                 'vo-v4-2-resnet-snapshot-2000-display-1-fulldens-batch-2_1514129205_snapshot-_iter_474000_threshold0';
%                 'vo-v4-2-resnet-catfeat-snapshot-2000-display-1-fulldens-batch-2_1514129183_snapshot-_iter_468000_threshold0';
% }
% model_list = {'vo-v4-2-resnet-dropout-snapshot-2000-display-1-dropout_fulldens-batch-2_1514857787_snapshot-_iter_26000_threshold0'}
% model_list = {'vo-v4-2-resnet-dropout-snapshot-2000-display-1-dropout_fulldens-batch-2_1514857787_snapshot-_iter_50000_threshold0'}
model_list = {
'vo-v4-2-resnet-dropout-snapshot-2000-display-1-dropout_fulldens-batch-2_1514857787_snapshot-_iter_26000_threshold0';
'xu_lstm';
% 'vo-v4-2-resnet-dropout-snapshot-2000-display-1-dropout_fulldens-batch-2_1514857787_snapshot-_iter_100000_threshold0';
% 'vo-v4-2-resnet-dropout-snapshot-2000-display-1-dropout_fulldens-batch-2_1514857787_snapshot-_iter_50000_threshold0';
}


cc_msk  = 1;
sim_msk = 1;
jud_msk = 1;
bor_msk = 1;
sauc_msk= 0;
emd_msk = 0;
kl_msk  = 1;
nss_msk = 1;
ig_msk  = 0;

dens_subdir_list = dir(dens_dir); % density/video1, video2....
fixa_subdir_list = dir(fixa_dir); % Fixation/video1, video2....
% baseline_map = imread('baseline_map.bmp');

for m = 1 : length(model_list)
    disp('hey')
    model_name = char(model_list(m));
    
    sal_dir = fullfile(base_sal_dir, model_name);   % _Saliencymap/SALICON
    sal_subdir_list = dir(sal_dir);                 % _Saliencymap/SALICON/video1, video2....
    save_mod_dir = fullfile(save_base_dir, model_name);   %_Result/SALICON
    if ~isdir(save_mod_dir)
        mkdir(save_mod_dir)
    end
    
    % for i = 3 : 12 %% skip . & .. , so start from 3
    for i = 3 : length(sal_subdir_list) %% skip . & .. , so start from 3
        disp('hey2')
        % save_dir = fullfile(save_mod_dir, sal_subdir_list(i).name);
        save_name = strcat(sal_subdir_list(i).name, '-', model_name, '.mat');
        save_path = fullfile(save_mod_dir, save_name);
        if exist(save_path)
            fprintf('%s exists. Skip~\n', save_path);
            continue
        end

        %%%% assert sal_subdir_list(i).name equals dens_subdir_list(i).name
        cur_sal_dir = fullfile(sal_dir, sal_subdir_list(i).name);
        cur_dens_dir = fullfile(dens_dir, dens_subdir_list(i).name);
        cur_fixa_dir = fullfile(fixa_dir, fixa_subdir_list(i).name);
        
        ext = {'*.jpg','*.bmp', '*.png', '*.jpeg'};
        saliencymap_path_list = [];
        densitymap_path_list = [];
        fixationmap_path_list = [];
        for e = 1:length(ext)
            saliencymap_path_list = [saliencymap_path_list dir(fullfile(cur_sal_dir, ext{e}))];
            densitymap_path_list = [densitymap_path_list dir(fullfile(cur_dens_dir, ext{e}))];
            fixationmap_path_list = [fixationmap_path_list dir(fullfile(cur_fixa_dir, ext{e}))];
        end
        saliencymap_path_list = natsortfiles({saliencymap_path_list.name});
        densitymap_path_list = natsortfiles({densitymap_path_list.name});
        fixationmap_path_list = natsortfiles({fixationmap_path_list.name});

        LengthFiles = length(fixationmap_path_list);
        true_length = LengthFiles - 2 * frame_cut
        saliency_score_CC = zeros(1,true_length);
        saliency_score_SIM = zeros(1,true_length);
        saliency_score_JUD = zeros(1,true_length);
        saliency_score_BOR = zeros(1,true_length);
        saliency_score_SAUC = zeros(1,true_length);
        saliency_score_EMD = zeros(1,true_length);
        saliency_score_KL = zeros(1,true_length);
        saliency_score_NSS = zeros(1,true_length);
        saliency_score_IG = zeros(1,true_length);
        
        %% CALCULATE METRICS %%
        disp('calculate the metrics...');
        t1=clock;
        % parfor j = 1+frame_cut : LengthFiles-frame_cut
        % for j = 1 : 2
        parfor j=1:true_length

            smap_path = char(fullfile(cur_sal_dir,saliencymap_path_list(j+frame_cut)))
            density_path = char(fullfile(cur_dens_dir,densitymap_path_list(j+frame_cut)))
            fixation_path = char(fullfile(cur_fixa_dir, fixationmap_path_list(j+frame_cut)))
            % jj = j-frame_cut;
            fprintf('Handling %s\n', smap_path);

            image_saliency = imread(smap_path);
            image_density = imread(density_path);
            image_fixation = imread(fixation_path);
            
            [row,col] = size(image_saliency);
            other_map=zeros(row, col);
            % imresize(image_saliency, size(image_density));

            if cc_msk
                %% CC %%
                % tic
                saliency_score_CC(j) = CC(image_saliency, image_density);
                % toc
            end
            
            if sim_msk
                %% SIM %% 
                % tic
                saliency_score_SIM(j) = similarity(image_saliency, image_density);
                % toc
            end
            
            if jud_msk
                %% AUCJUDD %%
                % tic
                % saliency_score_JUD(j) = AUC_Judd(image_saliency, image_density);
                saliency_score_JUD(j) = AUC_Judd(image_saliency, image_fixation);
                % toc
            end
            
            if bor_msk
                %% AUCBorji %%
                % tic
                % saliency_score_BOR(j) = AUC_Borji(image_saliency, image_density);
                saliency_score_BOR(j) = AUC_Borji(image_saliency, image_fixation);
                % toc
            end
            
            if sauc_msk
                %% SAUC %%
                % tic
                other_map = compute_othermap(10, row, col, all_in_one_fixation_directory);
                saliency_score_SAUC(j) = AUC_shuffled(image_saliency, image_fixation, other_map);
                % toc
            end
            
            if emd_msk
                %% EMD %%
                % tic
                saliency_score_EMD(j) = EMD(image_saliency, image_fixation);
                % saliency_score_EMD(j) = emd_hat_mex(image_saliency, image_fixation, image_fixation);
                % toc
            end
            
            if kl_msk
                %% KL %%
                % tic
                % saliency_score_KL(j) = KLdiv(image_saliency, image_density);
                saliency_score_KL(j) = KLdiv(image_saliency, image_fixation);
                % toc
            end
                
            if nss_msk
                %% NSS %%
                % tic
                % saliency_score_NSS(j)=NSS(image_saliency, image_density);
                saliency_score_NSS(j)=NSS(image_saliency, image_fixation);
                % toc
            end
            
            if ig_msk
                %% InfoGain %%
                % tic
                % saliency_score_NSS(j)=NSS(image_saliency, image_density);
                b_map = baseline_map;
                b_map(logical(image_fixation))=0;
                saliency_score_IG(j) = InfoGain(double(image_saliency), double(image_fixation), double(b_map));
                % saliency_score_IG(j)=InfoGain(image_saliency, image_fixation);
                % toc
            end
        end

        fprintf('Done for %s',sal_subdir_list(i).name)
        saliency_score=[saliency_score_CC;saliency_score_SIM;
                        saliency_score_JUD;saliency_score_BOR;
                        saliency_score_SAUC;saliency_score_EMD;
                        saliency_score_KL;saliency_score_NSS;
                        saliency_score_IG;]
        t2=clock;
        time_cost=etime(t2,t1);


        save(save_path, 'saliency_score','time_cost');
        fprintf('%s saved\n',save_path);
    end
    fprintf('Done for %s',model_name);
end


%  _______  _______  _______  _______  ___   _______  _______  ___   _______  _______ 
% |       ||       ||   _   ||       ||   | |       ||       ||   | |       ||       |
% |  _____||_     _||  |_|  ||_     _||   | |  _____||_     _||   | |       ||  _____|
% | |_____   |   |  |       |  |   |  |   | | |_____   |   |  |   | |       || |_____ 
% |_____  |  |   |  |       |  |   |  |   | |_____  |  |   |  |   | |      _||_____  |
%  _____| |  |   |  |   _   |  |   |  |   |  _____| |  |   |  |   | |     |_  _____| |
% |_______|  |___|  |__| |__|  |___|  |___| |_______|  |___|  |___| |_______||_______|
model_base_dir=base_dir;
result_base_dir=base_dir;

if ~isdir(result_base_dir)
    mkdir(result_base_dir);
end


met_count = 7;

for i=1:length(model_list)
    modelname=char(model_list(i));
    % disp(modelname);
    modeldir=fullfile(model_base_dir, modelname);
    
    vomat_list = dir(fullfile(modeldir, '*.mat*'));
    met_sum=zeros(met_count,1);
    frame_count=zeros(met_count,1);

    % cc=[];
    % sim=[];
    % jud=[];
    % bor=[];
    % kl=[];
    % nss=[];
    % total_metric=[cc;sim;jud;bor;kl;nss]
    for j=1:length(vomat_list)
        vomat_path=fullfile(modeldir, vomat_list(j).name);
        load(vomat_path);
        saliency_score([6], :) = []; % delete row 5&6, namely emd

        for k=1:met_count
            metric=saliency_score(k, :);

            [m,n]=find(isnan(metric)==1);
            % metric(:,m)=[]; % delete all nan
            metric(m,:)=[]; % delete all nan

            
            met_sum(k) = met_sum(k) + sum(metric(:));
            frame_count(k) = frame_count(k) + length(metric);
        end
    end
    
    result=zeros(met_count,1);
    for k=1:met_count
        result(k)=met_sum(k)/frame_count(k);
    end
    resultname=strcat(modelname,  '-result.mat');
    resultpath = fullfile(result_base_dir, resultname);
    save(resultpath, 'result','frame_count','met_sum');
    fprintf('%s saved!\n',resultpath);
    result'
end