import shutil
import os

snapshot_basedir = '/data/sunnycia/saliency_on_videoset/Train/training_output/salicon'
figure_basedir = '/data/sunnycia/saliency_on_videoset/Train/figure'

snapshot_folder_list = os.listdir(snapshot_basedir)
figure_folder_list = os.listdir(figure_basedir)

for snapshot_folder in snapshot_folder_list:
    path = os.path.join(snapshot_basedir, snapshot_folder)
    if os.listdir(path) == []:
        print snapshot_folder, "will be deleted"
        shutil.rmtree(path)
        print "\tDone!"
    else:
        print snapshot_folder, "won't be deleted"

for figure_folder in figure_folder_list:
    path = os.path.join(figure_basedir, figure_folder)
    if os.listdir(path) == []:
        print figure_folder, "will be deleted"
        shutil.rmtree(path)
        print "\tDone!"
    else:
        print figure_folder, "won't be deleted"
