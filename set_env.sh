#!/bin/bash 
export CUDA_VISIBLE_DEVICES=7
# CAFFE_DIR=../caffe-flownet
CAFFE_DIR=$1
export PYTHONPATH=$CAFFE_DIR/python:$PYTHONPATH

RELEASE_PATH="$CAFFE_DIR/build" 
export LD_LIBRARY_PATH="$RELEASE_PATH/lib:$LD_LIBRARY_PATH"
export PATH="$RELEASE_PATH/tools:$RELEASE_PATH/scripts:$PATH"
export CAFFE_BIN="$RELEASE_PATH/tools/caffe"

# export PYTHONPATH=../caffe-flownet/python:$PYTHONPATH


