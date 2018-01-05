%%%%%%%sss%%uu%%%uu%%nnnn%%%%nnnn%%%%yy%%%yy%%cccc%%iii%%%%aa%%%%%%%%
%%%%%sss%%%%uu%%%uu%nn%%%nn%nn%%%nn%%%yy%yy%cc%%%%%%iii%%%a%%a%%%%%%%
%%%%%%%%sss%uu%%%uu%nn%%%nn%nn%%%nn%%%%yy%%%cc%%%%%%iii%%aaaaaa%%%%%%
%%%%%sss%%%%%uuuu%%%nn%%%nn%nn%%%nn%%%%yy%%%%%cccc%%iii%aa%%%%aa%%%%%

%%% for dvi test
% dvi=6
% base_dir='/home/sunnycia/pwd/saliency_on_videoset/_Result_trans/_Result_trans_';
% model_base_dir=strcat(base_dir, num2str(dvi));
% result_base_dir=strcat(base_dir, num2str(dvi));

% 
% base_dir='/data/sunnycia/saliency_on_videoset/Train/metric-matlab/videoset';
% base_dir='/data/sunnycia/saliency_on_videoset/_Metric_results/nopar/final/_Result_1_fc_6';
% base_dir='/home/sunnycia/pwd/saliency_on_videoset/vt_result/_Result_1';
% base_dir='/home/sunnycia/pwd/saliency_on_videoset/novt_result/_Result_8';

% base_dir='/home/sunnycia/pwd/saliency_on_videoset/novt_result/BF_dropout/_Result_8';
% base_dir='/home/sunnycia/pwd/saliency_on_videoset/vt_result/_Result_Threshold8';
model_base_dir=base_dir;
result_base_dir=base_dir;

if ~isdir(result_base_dir)
    mkdir(result_base_dir);
end

% model_list={'v1';'v3';};
% model_list={'train_kldloss-kld_weight-100-batch-1_1510102029_usesnapshot_1509584263_snapshot-_iter_100000'};
% model_list = {'xu_dupext40', 'SAM', 'train_kldloss-kld_weight-100-batch-1_1510102029_usesnapshot_1509584263_snapshot-_iter_100000'}
% model_list = {'vo-v4-2-base_lr-0.01-snapshot-20000-display-1-batch-8_1513218849_snapshot-13500'};
% model_list = {'vo-v4-2-base_lr-0.01-snapshot-20000-display-1-batch-8_1513218849_snapshot-13500_threshold0.75_rangesmooth'}
% model_list = {'vo-v4-2-snapshot-999999-display-1-ledovSet-batch-8_1513764025_snapshot-11250_threshold0'}
% model_list = {'vo-v3-2_train_kldloss_withouteuc-batch-1_1513084718_snapshot-_iter_150000'};
% model_list = {'DENSITY';'MOTION';'MSFUSION';'UNCERTAINTY';'FANG2';'SUN';'MDB';'ISEEL';'SAM';'SALICON';'ITKO';'GBVS';'PQFT';'XU'};
% model_list={'MSFUSION';'UNCERTAINTY';};
% model_list = {'vo-v4-2-resnet-snapshot-2000-display-1--batch-2_1514034705_snapshot-_iter_72000_threshold0';
% 'vo-v4-2-snapshot-2000-display-1--batch-8_1514033989_snapshot-_iter_20000_threshold0';};
% model_list = {'vo-v4-2-resnet-catfeat-snapshot-2000-display-1--batch-2_1514034491_snapshot-_iter_72000_threshold0'};
% model_list = {'vo-v4-2-snapshot-2000-display-1-fulldens-batch-8_1514129167_snapshot-_iter_28000_threshold0'}
% model_list = {'vo-v4-2-resnet-base_lr-0.01-snapshot-2000-display-1--batch-2_1514260519_usesnapshot_1514034705_snapshot-_iter_72000_snapshot-_iter_96000_threshold0'}

% model_list = {  'vo-v4-2-resnet-base_lr-0.01-snapshot-2000-display-1--batch-2_1514260519_usesnapshot_1514034705_snapshot-_iter_72000_snapshot-_iter_454000_threshold0';
%                 'vo-v4-2-resnet-snapshot-2000-display-1-fulldens-batch-2_1514129205_snapshot-_iter_474000_threshold0';
%                 'vo-v4-2-resnet-catfeat-snapshot-2000-display-1-fulldens-batch-2_1514129183_snapshot-_iter_468000_threshold0';
% }
% model_list = {'vo-v4-2-resnet-dropout-snapshot-2000-display-1-dropout_fulldens-batch-2_1514857787_snapshot-_iter_26000_threshold0'}
% model_list = {%'xu_lstm';
% 'vo-v4-2-resnet-dropout-snapshot-2000-display-1-dropout_fulldens-batch-2_1514857787_snapshot-_iter_100000_threshold0';
% 'vo-v4-2-resnet-dropout-snapshot-2000-display-1-dropout_fulldens-batch-2_1514857787_snapshot-_iter_50000_threshold0';
% 'vo-v4-2-resnet-dropout-snapshot-2000-display-1-dropout_fulldens-batch-2_1514857787_snapshot-_iter_26000_threshold0';
% }




%{
    metric index reference
    1    2    3    4    5*   6*   7    8   
    cc   sim  jud  bor  sauc emd  kl   nss
%}

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
        % saliency_score([5, 6], :) = []; % delete row 5&6, namely sauc & emd
        saliency_score([6], :) = []; % delete row 5&6, namely emd
        % saliency_score
        % cc_list = saliency_score(1,:)
        % break;
        for k=1:met_count
            metric=saliency_score(k, :);
            %{
                code block for delete nan col
                a=[1 3 5; 2 NaN 6; 7 8 9; NaN NaN 7];
                [m,n]=find(isnan(a)==1);
                a(m,:)=[]
            %}
            
            [m,n]=find(isnan(metric)==1);
            % metric(:,m)=[]; % delete all nan
            metric(m,:)=[]; % delete all nan
            
            % append  metric
            % total_metric(k)=[total_metric(k) metric];
            % cc=[cc metric(1)];
            % sim=[sim metric(2)];
            % jud=[bor metric(3)];
            % bor=[bor metric(4)];
            % sauc=[sauc metric(5)];
            % kl=[kl metric(6)];
            % nss=[nss metric(6)];
            
            met_sum(k) = met_sum(k) + sum(metric(:));
            frame_count(k) = frame_count(k) + length(metric);
        end
    end
    
    result=zeros(met_count,1);
    for k=1:met_count
        result(k)=met_sum(k)/frame_count(k);
        % std_devi(k)=std(total_metric(k));
    end
    resultname=strcat(modelname,  '-result.mat');
    resultpath = fullfile(result_base_dir, resultname);
    save(resultpath, 'result','frame_count','met_sum');
    fprintf('%s saved!\n',resultpath);
    % result'
    
    % break;
end