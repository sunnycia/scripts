%%%%%%%sss%%uu%%%uu%%nnnn%%%%nnnn%%%%yy%%%yy%%cccc%%iii%%%%aa%%%%%%%%
%%%%%sss%%%%uu%%%uu%nn%%%nn%nn%%%nn%%%yy%yy%cc%%%%%%iii%%%a%%a%%%%%%%
%%%%%%%%sss%uu%%%uu%nn%%%nn%nn%%%nn%%%%yy%%%cc%%%%%%iii%%aaaaaa%%%%%%
%%%%%sss%%%%%uuuu%%%nn%%%nn%nn%%%nn%%%%yy%%%%%cccc%%iii%aa%%%%aa%%%%%

% Other map computer for videoset.
function other_map = compute_othermap(M, row,column,map_dir)
% addpath(map_dir)
fixation_list = dir(map_dir);
rdm_idx = [];
% rdm_cnt = 0;
while length(rdm_idx) < M
    cur_idx = floor(rand(1, 1)*(length(fixation_list)-3)) + 3;
    rdm_idx = [rdm_idx, cur_idx];
end
% rdm_idx
other_map = zeros(row, column);
for j = 1 : M
    idx = rdm_idx(j);
    % fixation_path = fullfile(map_dir, fixation_list(idx).name)
    % fixationmap = imread(fixation_path)
    load(fullfile(map_dir, fixation_list(idx).name))
    fixation = imresize(fixation, size(other_map));
    % size(fixation)
    % size(other_map)
    other_map(logical(fixation))=1;
end