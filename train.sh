#!/bin/sh

CAFFE_DIR=../C3D-v1.1-kldloss
GPU_ID=7
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTHONPATH=$CAFFE_DIR/python:$PYTHONPATH

RELEASE_PATH="$CAFFE_DIR/build" 
export LD_LIBRARY_PATH="$RELEASE_PATH/lib:$LD_LIBRARY_PATH"
export PATH="$RELEASE_PATH/tools:$RELEASE_PATH/scripts:$PATH"
export CAFFE_BIN="$RELEASE_PATH/tools/caffe"

MODEL=ns_v1_naive
LOSS=EuclideanLoss
DEBUG=1

BATCH=2
WIDTH=112
HEIGHT=112

PLOT_ITER=100
VALID_ITER=5000
SNAPSHOT_ITER=5000
TRAIN_PROPS=0.999
BASE_LR=0.01
LR_POLICY='inv'

CONNECTION=0
CLIP_LENGTH=16
OVERLAP=15

TRAIN_BASE='hollywood'

NETWORK=prototxt/$MODEL.prototxt
PRETRAINED_MODEL='../pretrained_model/c3d_resnet18_sports1m_r2_iter_2800000.caffemodel'
# SNAPSHOT_DIR=snashot/v1_basic/2018091321:47:38
MODEL_DIR=$MODEL

if [ -n "$DEBUG" ];
then
    MODEL_DIR=debug_$MODEL
fi

TS=`date "+%Y%m%d%T"`
if [ -n "$SNAPSHOT_DIR" ];
then
    SNAPSHOT_DIR=$(ls $SNAPSHOT_DIR/*solverstate -t1 |  head -n 1) # the latest solverstate
    SOLVER=$(ls $SNAPSHOT_DIR/*.prototxt -s1 |head -n 1) # the smaller prototxt
    NET=$(ls $SNAPSHOT_DIR/*.prototxt -S1 | head -n 1) # the bigger prototxt
else
    SNAPSHOT_DIR=../training_output/$MODEL_DIR/$TS
    SOLVER=prototxt/solver.prototxt
    NET=$NETWORK
fi 

python gen_network.py --network_path=$NET \
                      --clip_length=$CLIP_LENGTH \
                      --height=$HEIGHT \
                      --width=$WIDTH \
                      --batch=$BATCH \
                      --loss=$LOSS \
                      --model=$MODEL \

rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi

python gen_solver.py --network_path=$NET \
                     --solver_path=$SOLVER \
                     --snapshot_dir=$SNAPSHOT_DIR \
                     --snapshot_iter=$SNAPSHOT_ITER \
                     --base_lr=$BASE_LR \
                     --lr_policy=$LR_POLICY 

rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi

python training_video_voxel_based.py \
        --train_prototxt=$NETWORK \
        --solver_prototxt=$SOLVER \
        --pretrained_model=$PRETRAINED_MODEL \
        --plot_iter=$PLOT_ITER \
        --valid_iter=$VALID_ITER \
        --snapshot_dir=$SNAPSHOT_DIR \
        --snapshot_iter=$SNAPSHOT_ITER \
        --training_example_props=$TRAIN_PROPS \
        --dataset=$TRAIN_BASE \
        --clip_length=$CLIP_LENGTH \
        --overlap=$OVERLAP \
        --batch=$BATCH \
        --debug=$DEBUG

rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi