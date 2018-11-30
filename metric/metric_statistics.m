%  _______  _______  _______  _______  ___   _______  _______  ___   _______  _______ 
% |       ||       ||   _   ||       ||   | |       ||       ||   | |       ||       |
% |  _____||_     _||  |_|  ||_     _||   | |  _____||_     _||   | |       ||  _____|
% | |_____   |   |  |       |  |   |  |   | | |_____   |   |  |   | |       || |_____ 
% |_____  |  |   |  |       |  |   |  |   | |_____  |  |   |  |   | |      _||_____  |
%  _____| |  |   |  |   _   |  |   |  |   |  _____| |  |   |  |   | |     |_  _____| |
% |_______|  |___|  |__| |__|  |___|  |___| |_______|  |___|  |___| |_______||_______|

if ~exist('save_base', 'var')
    fprintf('save_base variable not exists');
end
if ~exist('model_name', 'var')
    fprintf('model_name variable not exists');
end

result_base_dir=save_base;

if ~isdir(result_base_dir)
    mkdir(result_base_dir);
end

met_count = 6;

vomat_list=dir(fullfile(save_base, '*.mat'));
met_sum=zeros(met_count,1);
frame_count=zeros(met_count,1);

for j=1:length(vomat_list)
    vomat_path=fullfile(save_base, vomat_list(j).name);
    load(vomat_path);
    % saliency_score([6], :) = []; % delete row 5&6, namely emd

    for k=1:met_count
        metric=saliency_score(k, :);

        [m,n]=find(isnan(metric)==1);
        metric(m,:)=[]; % delete all nan

        met_sum(k) = met_sum(k) + sum(metric(:));
        frame_count(k) = frame_count(k) + length(metric);
    end
end

result=zeros(met_count,1);
for k=1:met_count
    result(k)=met_sum(k)/frame_count(k);
end
resultname=strcat(model_name,  '-result.mat');
resultpath = fullfile(result_base_dir, resultname);
save(resultpath, 'result','frame_count','met_sum');
fprintf('%s saved!\n',resultpath);
result'