from math import ceil
import random
import cv2
import numpy as np
import glob
# from PIL import Image
import imghdr
import os
import random

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
def jigsaw(imageDir, output_path=None, stdsize=(30, 30), padding=0, mode=cv2.IMREAD_COLOR, random=False, rdm_portion=1, arrange=0, tall_img=False):
    imgs = []
    delta=0
    imageList = os.listdir(imageDir)
    if random:
        random.shuffle(imageList)
    else:
        imageList.sort()
    
    file_num = len(imageList)
    for filename in imageList:
        imagePath = os.path.join(imageDir, filename)
        ## check if an image
        imgType = imghdr.what(imagePath)
        if imgType==None:
            print imagePath, "is not an regular Image file"
            file_num -= 1
            continue
        
        img_arr = cv2.imread(imagePath, mode)
        img_arr = cv2.resize(img_arr, stdsize)
        imgs.append(img_arr)

    
    if arrange==0:
        row, col = explode_number(file_num)
        if tall_img:
            t = row;row=col;col=t;
    else:
        row = arrange[0]
        col = arrange[1]
    print row, col;
    if row*col > file_num:
        delta =  row*col - file_num
        if mode == cv2.IMREAD_GRAYSCALE:
            patch_img = np.ones(stdsize)
        else:
            patch_img = np.ones((stdsize[1], stdsize[0], 3))
        patch_img *= 255
        for i in range(delta):
            imgs.append(patch_img)
    print imgs[0].shape, imgs[-1].shape
    img = np.concatenate(imgs, 0)
    # cv2.imwrite("damn.jpg", img);exit()
    print img.shape
    if mode == cv2.IMREAD_GRAYSCALE:
        img = img.reshape(row, col, stdsize[0], stdsize[1])
        img = img.swapaxes(1, 2).reshape(row*stdsize[0], col*stdsize[1])
    else:
        img = img.reshape(row, col, stdsize[1], stdsize[0], 3)
        print img.shape
        if not padding==0:
            mask = np.ones(img.shape[:-1], dtype=bool)
            mask[:, :, padding:-padding, padding:-padding]=False
            img[mask] = 255
        
        img = img.swapaxes(1, 2).reshape(row*stdsize[1], col*stdsize[0], 3)
        print img.shape
    
    if output_path == None:
        cv2.imwrite(imageDir+".jpg", img)
    else:
        cv2.imwrite(output_path, img)