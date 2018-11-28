#!/bin/bash

GPU=7
export CUDA_VISIBLE_DEVICES=$GPU
CAFFE_DIR=../C3D-v1.1-tmp
export PYTHONPATH=$CAFFE_DIR/python:$PYTHONPATH

DATASET=gazecom
OVERLAP=8
SNAPSHOT_BASE=/data/sunnycia/saliency_on_videoset/Train/training_output/ns_v1_3dresnet18_pyramid/2018112511:01:47

python test_video.py --dataset=$DATASET --video_deploy_path=$SNAPSHOT_BASE/ns_v1_3dresnet18_pyramid.prototxt --video_model_path=$SNAPSHOT_BASE/snapshot__iter_50000.caffemodel --overlap=$OVERLAP

python test_video.py --dataset=$DATASET --video_deploy_path=$SNAPSHOT_BASE/ns_v1_3dresnet18_pyramid.prototxt --video_model_path=$SNAPSHOT_BASE/snapshot__iter_100000.caffemodel --overlap=$OVERLAP 

python test_video.py --dataset=$DATASET --video_deploy_path=$SNAPSHOT_BASE/ns_v1_3dresnet18_pyramid.prototxt --video_model_path=$SNAPSHOT_BASE/snapshot__iter_150000.caffemodel --overlap=$OVERLAP 

python test_video.py --dataset=$DATASET --video_deploy_path=$SNAPSHOT_BASE/ns_v1_3dresnet18_pyramid.prototxt --video_model_path=$SNAPSHOT_BASE/snapshot__iter_200000.caffemodel --overlap=$OVERLAP 


DATASET=diem
python test_video.py --dataset=$DATASET --video_deploy_path=$SNAPSHOT_BASE/ns_v1_3dresnet18_pyramid.prototxt --video_model_path=$SNAPSHOT_BASE/snapshot__iter_50000.caffemodel --overlap=$OVERLAP

python test_video.py --dataset=$DATASET --video_deploy_path=$SNAPSHOT_BASE/ns_v1_3dresnet18_pyramid.prototxt --video_model_path=$SNAPSHOT_BASE/snapshot__iter_100000.caffemodel --overlap=$OVERLAP 

python test_video.py --dataset=$DATASET --video_deploy_path=$SNAPSHOT_BASE/ns_v1_3dresnet18_pyramid.prototxt --video_model_path=$SNAPSHOT_BASE/snapshot__iter_150000.caffemodel --overlap=$OVERLAP 

python test_video.py --dataset=$DATASET --video_deploy_path=$SNAPSHOT_BASE/ns_v1_3dresnet18_pyramid.prototxt --video_model_path=$SNAPSHOT_BASE/snapshot__iter_200000.caffemodel --overlap=$OVERLAP 
