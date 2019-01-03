addpath(genpath('../code_forVisualization'));
addpath(genpath('/data/sunnycia/saliency_on_videoset/Train/scripts/metric/code4metric'));
% download code from https://github.com/cvzoya/saliency/tree/master/code_forMetrics
%addpath(genpath('/Users/Zoya/Dropbox (MIT)/saliencyPageFiles/saliencyCode/code_forMetrics/')); 

if ~exist('frame_path', 'var')
    fprintf('frame_path variable not exists\n');
end
if ~exist('density_path', 'var')
    fprintf('density_path variable not exists\n');
end
if ~exist('fixation_path', 'var')
    fprintf('fixation_path variable not exists\n');
end
if ~exist('saliency_path', 'var')
    fprintf('saliency_path variable not exists\n');
end

im = imread(frame_path);
density_map = imread(density_path); % ground truth human fixation map (continuous distribution)
fixation_map = imread(fixation_path);
salMap = imread(saliency_path);   % saliency map (continuous distribution)

%% plot ground truth
figure('name','Ground Truth'); subplot(1,5,1); imshow(im); title('Frame sample');
subplot(1,5,2); imshow(density_map); colormap('parula'); freezeColors;
title('Density map'); 
subplot(1,5,3); 
se = strel('disk',15);
fixLocs_dil = imdilate(fixation_map,se);
imshow(fixLocs_dil); colormap('gray'); freezeColors;
title('Fixation map');

npoints = 10; % number of points to sample on ROC curve

% prepare the color map for the correctly detected and missed fixations
colmap = fliplr(colormap(jet(npoints)));
G = linspace(0.5,1,20)';
tpmap = horzcat(zeros(size(G)),G,zeros(size(G))); % green color space
fpmap = horzcat(G,zeros(size(G)),zeros(size(G))); % red color space

% compute AUC-Judd
heatmap = im2double(salMap);
[score,tp,fp,allthreshes] = AUC_Judd(heatmap, fixation_map);

N = ceil(length(allthreshes)/npoints);
allthreshes_samp = allthreshes(1:N:end);

heatmap_norm = (heatmap-min(heatmap(:)))/(max(heatmap(:))-min(heatmap(:)));

salMap_col = makeLevelSets(heatmap, allthreshes_samp, colmap);
% figure; subplot(1,2,1); imshow(salMap_col);
subplot(1,5,4);
heatmap = im2double(salMap);
heatmap_norm = (heatmap-min(heatmap(:)))/(max(heatmap(:))-min(heatmap(:)));
salMap_col = makeLevelSets(heatmap, allthreshes_samp, colmap);
imshow(salMap_col);
title('Saliency map');

% plot the ROC curve
tp1 = tp(1:N:end); fp1 = fp(1:N:end);
subplot(1,5,5); plot(fp,tp,'b'); hold on;
for ii = 1:npoints
     plot(fp1(ii),tp1(ii),'.','color',colmap(ii,:),'markersize',20); hold on; axis square;
end
title(sprintf('AUC: %2.2f',score),'fontsize',14);
xlabel('FP rate','fontsize',14); ylabel('TP rate','fontsize',14);

% set(gca,'LooseInset',get(gca,'TightInset'))
saveas(gcf,'roc.png');
close(figure);

% % plot the level sets, one per subplot
% figure('name','saliency map level sets');
% nplot = floor(npoints/2); % plot every other level set
% for ii = 1:nplot
%    %temp = heatmap_norm>=allthreshes_samp(ii);   % plot every level set
%    temp = heatmap_norm>=allthreshes_samp(2*ii);  % plot every other level set
%    temp2 = zeros(size(temp,1),size(temp,2),3);
%    temp2(:,:,1) = temp*colmap(2*ii,1); temp2(:,:,2) = temp*colmap(2*ii,2); temp2(:,:,3) = temp*colmap(2*ii,3); 
%    subplottight(1,nplot,ii,0.05); imshow(temp2);
% end

% set(gca,'LooseInset',get(gca,'TightInset'))
% saveas(gcf,'level_set.png');
% close(figure);
% close(gcf)


figure
imshow(im)
set(gca,'units','centimeters')
pos = get(gca,'Position');
ti = get(gca,'TightInset');

ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
saveas(gcf,'frame.png');
close(figure);
close(gcf)

imshow(density_map); colormap('parula'); freezeColors;
set(gca,'units','centimeters')
pos = get(gca,'Position');
ti = get(gca,'TightInset');

ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
saveas(gcf,'density.png');
close(figure);
close(gcf)

imshow(fixLocs_dil)
set(gca,'units','centimeters')
pos = get(gca,'Position');
ti = get(gca,'TightInset');

ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
saveas(gcf,'fixation.png');
close(figure);
close(gcf)

imshow(salMap_col)
set(gca,'units','centimeters')
pos = get(gca,'Position');
ti = get(gca,'TightInset');

ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
saveas(gcf,'saliency.png');
close(figure);
close(gcf)


tp1 = tp(1:N:end); fp1 = fp(1:N:end);
plot(fp,tp,'b','Linewidth',2); hold on;
for ii = 1:npoints
     plot(fp1(ii),tp1(ii),'.','color',colmap(ii,:),'markersize',25); hold on; axis square;
end
set(gca,'FontSize',15)
fontsize=20;
title(sprintf('AUC: %2.2f',score),'fontsize',fontsize);
xlabel('FP rate','fontsize',fontsize,'FontName', 'Times New Roman'); ylabel('TP rate','fontsize',fontsize,'FontName', 'Times New Roman');

set(gca,'units','centimeters')
pos = get(gca,'Position');
ti = get(gca,'TightInset');

ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
saveas(gcf,'roc.png');
close(figure);
close(gcf)
