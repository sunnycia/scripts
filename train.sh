#!/bin/sh

# PROTO=train_nss-kldloss_withouteuc

PROTO=$1
debug=$2
pre=${PROTO##*/}
pre=${pre%.*}
LOG_PATH='../log/'$pre
echo $PROTO
echo debug=$debug
echo $LOG_PATH
python training.py --debug=$debug --train_prototxt=$PROTO 2>&1 | tee $LOG_PATH
