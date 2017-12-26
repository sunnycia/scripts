# -*- coding: UTF-8 -*-
import commands
import os
import re# -*- coding: UTF-8 -*-
import commands
import os
import re

caffe_root = '/home/yehs/caffe-master/'
path = "/home/yehs/train_module/"
images_path = path + "dataset/"
def create_db(txt_save_path):
    #lmdb文件名字
    lmdb_name = 'train.lmdb'
    #生成的db文件的保存目录
    lmdb_save_path = path  +  lmdb_name
    #convert_imageset工具路径
    convert_imageset_path = caffe_root + 'build/tools/convert_imageset'
    cmd = """%s --shuffle --resize_height=128 --resize_width=128 %s %s %s"""
    status, output = commands.getstatusoutput(cmd % (convert_imageset_path, images_path,
        txt_save_path, lmdb_save_path))
    print output
    if(status == 0):
        print "lmbd文件生成成功"

if __name__  == '__main__':
  txt_save_path = path + "train.txt"
  create_db(txt_save_path)
  #txt_save_path = path + "test.txt"
  #create_db(txt_save_path) 

