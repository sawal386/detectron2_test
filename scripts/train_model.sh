#!/bin/bash

#  train_model.sh
#  
#
#  Created by Sawal Acharya on 1/6/23.
#  

#!/bin/bash

python main_train.py \
    --model_path "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml" \
    --train_data_catalog_name "balloon_train" \
    --test_data_name "balloon_val" \
    --batch_size 128 \
    --im_per_batch 2 \
    --n_class 1 \
    --output_dir_path "outputs/mask_rcnn_test"\
    --learning_rate 0.001 \
    --n_iters 300 \
    --raw_data_loc "data/balloon/train" \
    --dataset_name "balloon"\
    --dataset_type "train" \
    --annotation_info "data/balloon/train/via_region_data.json"
