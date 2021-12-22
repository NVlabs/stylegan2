#!/bin/bash
# MSOE ROSIE run StyleGAN2
# author: gagany <daroachgb@msoe.edu>

set MCW_RESEARCH=/srv/data/mcw_research
set TRAINING_DATA=$MCW_RESEARCH/tfrecord/0.5x
set OUTPUT=$MCW_RESEARCH/stylegan2/training_output

python run_training.py --num-gpus=8 --data-dir=$OUTPUT --config=config-f --dataset=$TRAINING_DATA --mirror-augment=false