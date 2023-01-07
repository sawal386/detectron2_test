#!/bin/bash 

python main.py \
    --model_path "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" \
    --test_image_loc "data/simple_test_images" \
    --test_image_names "test_image1.png"
