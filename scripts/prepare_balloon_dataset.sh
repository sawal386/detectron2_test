#!/bin/sh

#  prepare_balloon_dataset.sh
#  
#
#  Created by Sawal Acharya on 1/6/23.
#

python prepare_data.py \
    --raw_data_loc "data/balloon/train" \
    --save_loc "data/balloon_detectron" \
    --dataset_name "balloon"\
    --dataset_type "train" \
    --annotation_info "data/balloon/train/via_region_data.json"

python prepare_data.py \
    --raw_data_loc "data/balloon/val" \
    --save_loc "data/balloon_detectron" \
    --dataset_name "balloon"\
    --dataset_type "val" \
    --annotation_info "data/balloon/val/via_region_data.json"
