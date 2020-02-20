#!/bin/env bash

data_root="./../dataset_rem_lr"

python3 model.py $data_root 200
python3 model.py $data_root 100
python3 model.py $data_root 50
python3 model.py $data_root 25
