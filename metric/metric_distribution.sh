#!/bin/bash

DIR=$1

python metric/metric_distribution.py --metricdir="$DIR" --outputdir='../metric_distribution' --metricname='cc'
python metric/metric_distribution.py --metricdir="$DIR" --outputdir='../metric_distribution' --metricname='sim'
python metric/metric_distribution.py --metricdir="$DIR" --outputdir='../metric_distribution' --metricname='auc_jud'
python metric/metric_distribution.py --metricdir="$DIR" --outputdir='../metric_distribution' --metricname='auc_bor'
python metric/metric_distribution.py --metricdir="$DIR" --outputdir='../metric_distribution' --metricname='sauc'
