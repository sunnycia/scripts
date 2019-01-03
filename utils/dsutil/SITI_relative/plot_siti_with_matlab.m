clc;
clear;

mat_dir= 'siti_mat';
save_dir = 'siti_scatter_plot_matlab'

mat_name_list = dir(mat_dir);
total_number = length(mat_name_list);
for i=3:length(mat_name_list)
    [filepath, name, ext] = fileparts(mat_name_list(i).name);
    name = strsplit(name, '_');
    name = upper(name(1));
    load(fullfile(mat_dir, mat_name_list(i).name));
    save_path = char(fullfile(save_dir, strcat(name, '.png')))
    % figure(i);
    scatter(SI,TI, 50, 'x', 'LineWidth', 3);
    title(name)
    xlabel('SI')
    ylabel('TI')
    set(gca,'FontSize',20)
    saveas(gcf,save_path,'png');
    % close(figure(i));
end