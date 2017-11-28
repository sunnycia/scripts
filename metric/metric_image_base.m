%  delete(gcp);
% matlabpool 8

metricsFolder = 'saliency/code_forMetrics'
addpath(genpath(metricsFolder))
dsname = 'mit1003'

save_base = '/data/sunnycia/saliency_on_videoset/Train/metric-matlab';
if ~isdir(save_base)
    mkdir(save_base);
end
model_list = {'cb-train_kldloss-kld_weight-100-batch-1_1510102029_usesnapshot_1509584263_snapshot-_iter_100000';'train_kldloss-kld_weight-100-batch-1_1510102029_usesnapshot_1509584263_snapshot-_iter_100000';}

if strcmp(dsname,'mit1003')==1
    sal_base = '/data/sunnycia/SaliencyDataset/Image/MIT1003/saliency';
    dens_dir = strcat('/data/sunnycia/SaliencyDataset/Image/MIT1003/ALLFIXATIONMAPS');
    fixa_dir = strcat('/data/sunnycia/SaliencyDataset/Image/MIT1003/fixPts');
    all_in_one_fixation_directory = '/data/sunnycia/SaliencyDataset/Image/MIT1003/fixPts';
end
if strcmp(dsname,'nus')==1
    % sal_base = '/data/sunnycia/SaliencyDataset/Image/MIT1003/saliency';
    % dens_dir = strcat('/data/sunnycia/SaliencyDataset/Image/MIT1003/ALLFIXATIONMAPS');
    % fixa_dir = strcat('/data/sunnycia/SaliencyDataset/Image/MIT1003/fixPts');
    % all_in_one_fixation_directory = '/data/sunnycia/SaliencyDataset/Image/MIT1003/fixPts';
end
if strcmp(dsname,'nctu')==1
    % sal_base = '/data/sunnycia/SaliencyDataset/Image/MIT1003/saliency';
    % dens_dir = strcat('/data/sunnycia/SaliencyDataset/Image/MIT1003/ALLFIXATIONMAPS');
    % fixa_dir = strcat('/data/sunnycia/SaliencyDataset/Image/MIT1003/fixPts');
    % all_in_one_fixation_directory = '/data/sunnycia/SaliencyDataset/Image/MIT1003/fixPts';
end

cc_msk  = 1;
sim_msk = 1;
jud_msk = 1;
bor_msk = 1;
sauc_msk= 1;
emd_msk = 0;
kl_msk  = 1;
nss_msk = 1;
ig_msk  = 0;

for m = 1 : length(model_list)
    disp('hey')
    model_name = char(model_list(m));
    
    sal_dir = fullfile(sal_base, model_name);

    ext = {'*.jpg','*.bmp', '*.png', '*.jpeg'};
    saliencymap_path_list = [];
    densitymap_path_list = [];
    fixationmap_path_list = [];
    for e = 1:length(ext)
        saliencymap_path_list = [saliencymap_path_list dir(fullfile(sal_dir, ext{e}))];
        densitymap_path_list = [densitymap_path_list dir(fullfile(dens_dir, ext{e}))];
        fixationmap_path_list = [fixationmap_path_list dir(fullfile(fixa_dir, ext{e}))];
    end
    
    saliencymap_path_list = natsortfiles({saliencymap_path_list.name})
    densitymap_path_list = natsortfiles({densitymap_path_list.name});
    fixationmap_path_list = natsortfiles({fixationmap_path_list.name});

    LengthFiles = length(fixationmap_path_list);
    saliency_score_CC = zeros(1,LengthFiles);saliency_score_SIM = zeros(1,LengthFiles);saliency_score_JUD = zeros(1,LengthFiles);
    saliency_score_BOR = zeros(1,LengthFiles);saliency_score_SAUC = zeros(1,LengthFiles);saliency_score_EMD = zeros(1,LengthFiles);
    saliency_score_KL = zeros(1,LengthFiles);saliency_score_NSS = zeros(1,LengthFiles);saliency_score_IG = zeros(1,LengthFiles);
    
    %% CALCULATE METRICS %%
    disp('calculate the metrics...');
    t1=clock;
    for j = 1 : LengthFiles
        smap_path = char(fullfile(sal_dir,saliencymap_path_list(j)));
        density_path = char(fullfile(dens_dir,densitymap_path_list(j)));
        fixation_path = char(fullfile(fixa_dir, fixationmap_path_list(j)));
        
        fprintf('Handling %s', smap_path);

        image_saliency = imread(smap_path);
        image_density = imread(density_path);
        image_fixation = imread(fixation_path);
        other_map=zeros(1080, 1920);
        
        imresize(image_saliency, size(image_density));

        if cc_msk
            %% CC %%
            saliency_score_CC(j) = CC(image_saliency, image_density);
        end
        
        if sim_msk
            %% SIM %% 
            saliency_score_SIM(j) = similarity(image_saliency, image_density);
        end
        
        if jud_msk
            %% AUCJUDD %%
            saliency_score_JUD(j) = AUC_Judd(image_saliency, image_fixation);
        end
        
        if bor_msk
            %% AUCBorji %%
            saliency_score_BOR(j) = AUC_Borji(image_saliency, image_fixation);
        end
        
        if sauc_msk
            %% SAUC %%
            other_map = compute_othermap(10, 1080, 1920, all_in_one_fixation_directory);
            saliency_score_SAUC(j) = AUC_shuffled(image_saliency, image_fixation, other_map);
        end
        
        if emd_msk
            %% EMD %%
            saliency_score_EMD(j) = EMD(image_saliency, image_fixation);
            % saliency_score_EMD(j) = emd_hat_mex(image_saliency, image_fixation, image_fixation);
        end
        
        if kl_msk
            %% KL %%
            saliency_score_KL(j) = KLdiv(image_saliency, image_fixation);
        end
            
        if nss_msk
            saliency_score_NSS(j)=NSS(image_saliency, image_fixation);
        end
        
        if ig_msk
            %% InfoGain %%
            b_map = baseline_map;
            b_map(logical(image_fixation))=0;
            saliency_score_IG(j) = InfoGain(double(image_saliency), double(image_fixation), double(b_map));
        end
        saliency_score=[saliency_score_CC;saliency_score_SIM;
                        saliency_score_JUD;saliency_score_BOR;
                        saliency_score_SAUC;saliency_score_EMD;
                        saliency_score_KL;saliency_score_NSS;
                        saliency_score_IG;];
        t2=clock;
        time_cost=etime(t2,t1);

        save_path = fullfile(save_base, strcat(dsname, model_name,'.mat'))
        save(save_path, 'saliency_score','time_cost');
        fprintf('%s saved\n',save_path);
    end
end