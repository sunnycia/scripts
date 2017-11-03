#!/bin/bash 
export CUDA_VISIBLE_DEVICES=6 
export PYTHONPATH=../caffe-flownet/python:$PYTHONPATH
RELEASE_PATH="../caffe-flownet/build" 
export LD_LIBRARY_PATH="$RELEASE_PATH/lib:$LD_LIBRARY_PATH"
export PATH="$RELEASE_PATH/tools:$RELEASE_PATH/scripts:$PATH"
export CAFFE_BIN="$RELEASE_PATH/tools/caffe"

# export PYTHONPATH=../caffe-flownet/python:$PYTHONPATH


