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
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--snapshot', type=bool, default=False)
parser.add_argument('--figure', type=bool, default=True)
args = parser.parse_args()



snapshot_basedir = '/data/sunnycia/saliency_on_videoset/Train/training_output/salicon'
figure_basedir = '/data/sunnycia/saliency_on_videoset/Train/figure'

snapshot_folder_list = os.listdir(snapshot_basedir)
figure_folder_list = os.listdir(figure_basedir)

if args.snapshot:
  for snapshot_folder in snapshot_folder_list:
      path = os.path.join(snapshot_basedir, snapshot_folder)
      if os.listdir(path) == []:
          print snapshot_folder, "will be deleted"
          shutil.rmtree(path)
          print "\tDone!"
      else:
          print snapshot_folder, "won't be deleted"

if args.figure:
  for figure_folder in figure_folder_list:
      path = os.path.join(figure_basedir, figure_folder)
      if os.listdir(path) == [] or len(os.listdir(path)) < 10:
          print figure_folder, "will be deleted"
          shutil.rmtree(path)
          print "\tDone!"
      else:
          print figure_folder, "won't be deleted"
