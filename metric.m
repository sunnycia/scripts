% delete(gcp)
% matlabpool 4
% % delete(gcp('nocreate'))
% myPool = parpool(2);
%% calculate the metrics of models 
% currentFolder = pwd;
metricsFolder = 'code4metric'
addpath(genpath(metricsFolder))

% folder_list={'snapshot-train_kldloss_iter_850000', 'snapshot-train_kldloss_withouteuc_iter_150000'}
folder_list={'snapshot-train_nss-kldloss_withouteuc_iter_150000', 'snapshot-train_nssloss_iter_550000', 'snapshot-train_nssloss_withouteuc_iter_100000'}

dens_dir = '/data/sunnycia/SaliencyDataset/Image/SALICON/DATA/train_val/val2014/density';
fixa_dir = '/data/sunnycia/SaliencyDataset/Image/SALICON/DATA/train_val/val2014/fixation';

for i=1:length(folder_list)
    folder = folder_list(i)

    sal_dir = char(fullfile('/data/sunnycia/SaliencyDataset/Image/SALICON/DATA/train_val/val2014/saliency/', folder))
    save_base_dir = '/data/sunnycia/saliency_on_videoset/Train/metric';

    s = dir(fullfile(sal_dir, '*.jpg'));
    d = dir(fullfile(dens_dir, '*.jpg'));
    f = dir(fullfile(fixa_dir, '*.mat'));

    saliencymap_path_list = {s.name};
    densitymap_path_list = {d.name};
    fixationmap_path_list = {f.name};
    saliencymap_path_list = sort(saliencymap_path_list);
    densitymap_path_list = sort(densitymap_path_list);
    fixationmap_path_list = sort(fixationmap_path_list);
    % disp(saliencymap_path_list);

    LengthFiles = length(saliencymap_path_list);
    saliency_score_CC = zeros(1,LengthFiles);
    saliency_score_SIM = zeros(1,LengthFiles);
    saliency_score_JUD = zeros(1,LengthFiles);
    saliency_score_BOR = zeros(1,LengthFiles);
    saliency_score_SAUC = zeros(1,LengthFiles);
    saliency_score_EMD = zeros(1,LengthFiles);
    saliency_score_KL = zeros(1,LengthFiles);
    saliency_score_NSS = zeros(1,LengthFiles);

    for j = 1 : LengthFiles
        sal_map_path = char(saliencymap_path_list(j))
        dens_map_path = char(densitymap_path_list(j))
        fix_map_path = char(fixationmap_path_list(j))
        [pathstr,sname,ext] = fileparts(sal_map_path);
        [pathstr,dname,ext] = fileparts(dens_map_path);
        [pathstr,fname,ext] = fileparts(fix_map_path);

        % sname, dname, fname;
        assert( strcmp(sname, dname)==1 && strcmp(dname, fname)==1)

        smap_path = fullfile(sal_dir, sal_map_path);
        density_path = fullfile(dens_dir, dens_map_path);
        fixation_path = fullfile(fixa_dir, fix_map_path);



        image_saliency = imread(smap_path);
        image_density = imread(density_path);
        load(fixation_path);
        image_fixation = fixation;
        

        %% CC %%
        %tic
        saliency_score_CC(j) = CC(image_saliency, image_density);
        %toc
        %% SIM %% 
        %tic
        saliency_score_SIM(j) = similarity(image_saliency, image_density);
        %toc
        
        %% AUCJUDD %%
        %tic
        saliency_score_JUD(j) = AUC_Judd(image_saliency, image_fixation, 0);
        %toc
        
        %% AUCBorji %%
        %tic
        saliency_score_BOR(j) = AUC_Borji(image_saliency, image_fixation);
        %toc
        
        %% SAUC %%
        %tic
        % saliency_score_SAUC(j) = AUC_shuffled(image_saliency, image_fixation, other_map);
        %toc
        
        %% EMD %%
        %tic
        % saliency_score_EMD(j) = EMD(image_saliency, image_density);
        % saliency_score_EMD(j) = emd_hat_mex(image_saliency, image_fixation, image_fixation);
        %toc
        %% KL %%
        %tic
        saliency_score_KL(j) = KLdiv(image_saliency, image_density);
        %toc
        
        %% NSS %%
        %tic
        saliency_score_NSS(j)=NSS(image_saliency, image_density);
        %toc
        % fprintf('Done for %s\n', smap_path);
    end
    saliency_score=[saliency_score_CC;saliency_score_SIM;
                    saliency_score_JUD;saliency_score_BOR;
                    saliency_score_SAUC;saliency_score_EMD;
                    saliency_score_KL;saliency_score_NSS;];

    % [pathstr,modelname,ext] = fileparts(sal_dir);
    save_name = strcat(folder, '.mat');
    save_path = fullfile(save_base_dir, save_name);
    save(char(save_path), 'saliency_score');
    fprintf('%s saved\n',save_path);

end