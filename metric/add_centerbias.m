% delete(gcp);
% matlabpool 8

% salmap_dir = '/data/sunnycia/SaliencyDataset/Image/MIT1003/saliency/train_kldloss-kld_weight-100-batch-1_1510102029_usesnapshot_1509584263_snapshot-_iter_100000';
salmap_dir = '/data/sunnycia/SaliencyDataset/Image/MIT300/saliency/train_kldloss-kld_weight-100-batch-1_1510102029_usesnapshot_1509584263_snapshot-_iter_100000';
% newmap_dir = '/data/sunnycia/SaliencyDataset/Image/MIT1003/saliency/cb-train_kldloss-kld_weight-100-batch-1_1510102029_usesnapshot_1509584263_snapshot-_iter_100000';
newmap_dir = '/data/sunnycia/SaliencyDataset/Image/MIT300/saliency/cb-train_kldloss-kld_weight-100-batch-1_1510102029_usesnapshot_1509584263_snapshot-_iter_100000';
mkdir(newmap_dir)

salmap_list = dir(fullfile(salmap_dir,'*.*'));

for i = 1 : length(salmap_list)
    if strcmp(salmap_list(i).name,'.')==1
        continue
    elseif strcmp(salmap_list(i).name,'..')==1
        continue
    end
    salmap_path = fullfile(salmap_dir, salmap_list(i).name);
    salmap = imread(salmap_path);

    salmap = center_bias(salmap);

    new_path = fullfile(newmap_dir, salmap_list(i).name);
    imwrite(salmap, new_path)
end