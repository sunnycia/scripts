#coding=utf-8
'''
  ██████  ▄████▄   ▄▄▄       ██▀███   ██▒   █▓▓█████  ███▄    █   ▄████ ▓█████  ██▀███  
▒██    ▒ ▒██▀ ▀█  ▒████▄    ▓██ ▒ ██▒▓██░   █▒▓█   ▀  ██ ▀█   █  ██▒ ▀█▒▓█   ▀ ▓██ ▒ ██▒
░ ▓██▄   ▒▓█    ▄ ▒██  ▀█▄  ▓██ ░▄█ ▒ ▓██  █▒░▒███   ▓██  ▀█ ██▒▒██░▄▄▄░▒███   ▓██ ░▄█ ▒
  ▒   ██▒▒▓▓▄ ▄██▒░██▄▄▄▄██ ▒██▀▀█▄    ▒██ █░░▒▓█  ▄ ▓██▒  ▐▌██▒░▓█  ██▓▒▓█  ▄ ▒██▀▀█▄  
▒██████▒▒▒ ▓███▀ ░ ▓█   ▓██▒░██▓ ▒██▒   ▒▀█░  ░▒████▒▒██░   ▓██░░▒▓███▀▒░▒████▒░██▓ ▒██▒
▒ ▒▓▒ ▒ ░░ ░▒ ▒  ░ ▒▒   ▓▒█░░ ▒▓ ░▒▓░   ░ ▐░  ░░ ▒░ ░░ ▒░   ▒ ▒  ░▒   ▒ ░░ ▒░ ░░ ▒▓ ░▒▓░
░ ░▒  ░ ░  ░  ▒     ▒   ▒▒ ░  ░▒ ░ ▒░   ░ ░░   ░ ░  ░░ ░░   ░ ▒░  ░   ░  ░ ░  ░  ░▒ ░ ▒░
░  ░  ░  ░          ░   ▒     ░░   ░      ░░     ░      ░   ░ ░ ░ ░   ░    ░     ░░   ░ 
      ░  ░ ░            ░  ░   ░           ░     ░  ░         ░       ░    ░  ░   ░     
         ░                                ░                                             
'''
import shutil
import os, glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--snapshot', type=bool, default=False)
parser.add_argument('--figure', type=bool, default=True)
parser.add_argument('--delete_list', type=bool, default=False)
args = parser.parse_args()

snapshot_basedir = '/data/sunnycia/saliency_on_videoset/Train/training_output/salicon'
figure_basedir = '/data/sunnycia/saliency_on_videoset/Train/figure'

snapshot_folder_list = os.listdir(snapshot_basedir)
figure_folder_list = os.listdir(figure_basedir)

def snapshot_sort(path):
  basename = os.path.basename(path)
  iter_index = basename.split('.')[0].split('_')[-1]
  return int(iter_index)

if args.snapshot:
  # w_f = open('delete_list.txt', 'w')
  for snapshot_folder in snapshot_folder_list:
      path = os.path.join(snapshot_basedir, snapshot_folder)
      if os.listdir(path) == []:
          print snapshot_folder, "will be deleted"
          shutil.rmtree(path)
          print "\tDone!"
      else:
          model_path_list = glob.glob(os.path.join(path, '*.caffemodel'))
          model_path_list.sort(key=lambda x: os.path.getmtime(x))
          state_path_list = glob.glob(os.path.join(path, '*.solverstate'))
          state_path_list.sort(key=lambda x: os.path.getmtime(x))
          # files.sort()

          # print model_path_list
          
          for i in range(len(model_path_list)):
            if i==0 or i==len(model_path_list)-1 or i==(len(model_path_list)-1)/2:
              continue
            os.remove(model_path_list[i])
            # print >> w_f, model_path_list[i]
          for i in range(len(state_path_list)):
            if i==0 or i==len(state_path_list)-1 or i==(len(state_path_list)-1)/2:
              continue
            os.remove(state_path_list[i])
            # print >> w_f, state_path_list[i]

          # print 'check_out delete_list.txt'
          # input()
          # print len(model_path_list);exit()
          print snapshot_folder, "won't be deleted"
    # w_f.close()

if args.figure:
  for figure_folder in figure_folder_list:
      path = os.path.join(figure_basedir, figure_folder)
      if os.listdir(path) == [] or len(os.listdir(path)) < 10:
          print figure_folder, "will be deleted"
          shutil.rmtree(path)
          print "\tDone!"
      else:
          print figure_folder, "won't be deleted"