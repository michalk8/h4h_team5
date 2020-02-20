#!/bin/env bash

data_root="./../dataset_rem_lr"
size=400

python3 main.py $data_root $size bilinear
python3 main.py $data_root $size bicubic
python3 main.py $data_root $size area
python3 main.py $data_root $size neighbor

