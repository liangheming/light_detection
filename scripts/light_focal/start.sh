#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate pytorch
nohup python train.py >> info.log 2>&1 &
