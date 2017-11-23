import numpy as np
import os
import logging
import time


def mean_without_nan(score_list):
    tmp_list = []
    for score in score_list:
        if not np.isnan(score):
            tmp_list.append(score)
    return np.mean(tmp_list)


def tic():
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()
    return startTime_for_tictoc

def toc():
    if 'startTime_for_tictoc' in globals():
        endTime = time.time()
        return endTime - startTime_for_tictoc
    else:
        return None

def check_prime(number): 
    # if not type(number)==int:
        # print "Please input an interger number."
        # return 0;
    ceil = int(np.sqrt(number));
    for i in range(2, ceil+1):
        if number%i == 0:
            return 0
    return 1
            
def explode_number(number):
    if not type(number)==int:
        print "Please input an interger number."
        return 0;
    while check_prime(number):
        print "It's a prime"
        number += 1
    a = int(np.sqrt(number))
    if a**2 == number:
        return a, a
    while not number%a == 0:
        a -= 1
    b = number /a
    if a > b:
        b, a = a, b
    return a, b
    
#coding:gbk 
class Colored(object):  
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    FUCHSIA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'      
  
    #: no color  
    RESET = '\033[0m'
  
    def color_str(self, color, s):  
        return '{}{}{}'.format(  
            getattr(self, color),  
            s,  
            self.RESET  
        )  
  
    def red(self, s):  
        return self.color_str('RED', s)  
  
    def green(self, s):  
        return self.color_str('GREEN', s)  
  
    def yellow(self, s):  
        return self.color_str('YELLOW', s)  
  
    def blue(self, s):  
        return self.color_str('BLUE', s)  
  
    def fuchsia(self, s):  
        return self.color_str('FUCHSIA', s)  
  
    def cyan(self, s):  
        return self.color_str('CYAN', s)  
  
    def white(self, s):  
        return self.color_str('WHITE', s)  

def create_logger(root_output_path, cfg, image_set):
# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao
# --------------------------------------------------------
    # set up logger
    if not os.path.exists(root_output_path):
        os.makedirs(root_output_path)
    assert os.path.exists(root_output_path), '{} does not exist'.format(root_output_path)

    cfg_name = os.path.basename(cfg).split('.')[0]
    config_output_path = os.path.join(root_output_path, '{}'.format(cfg_name))
    if not os.path.exists(config_output_path):
        os.makedirs(config_output_path)

    image_sets = [iset for iset in image_set.split('+')]
    final_output_path = os.path.join(config_output_path, '{}'.format('_'.join(image_sets)))
    if not os.path.exists(final_output_path):
        os.makedirs(final_output_path)

    log_file = '{}_{}.log'.format(cfg_name, time.strftime('%Y-%m-%d-%H-%M'))
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(final_output_path, log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    return logger, final_output_path

