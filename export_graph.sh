#/usr/bin/bash

MODEL_DIR=ssd_mobilenet_v2_fpnlite_papsmear-v2
PIPELINE_CONFIG_PATH=configs/${MODEL_DIR}.config
#PIPELINE_CONFIG_PATH=configs/centernet_resnet101_512x512_coco17_tpu8.config
# PIPELINE_CONFIG_PATH=configs/centernet_mobileent_v2_fpn_512x512.config
CHECKPOINT_DIR=training/scale_15_overlap7/
#CHECKPOINT_DIR=backup/training/
OUTPUT_DIR=model_exported

# clear old exported model
rm -rf ${OUTPUT_DIR}

PYTHONPATH=`pwd`/models/research:`pwd`/models/research/slim \
TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=0 \
    python3 ./models/research/object_detection/export_tflite_graph_tf2.py \
            --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
            --trained_checkpoint_dir=${CHECKPOINT_DIR} \
            --output_directory=${OUTPUT_DIR} \
            --max_detections=100


#PYTHONPATH=`pwd`/models/research:`pwd`/models/research/slim \
#TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=2,3 CUDA_VISIBLE_DEVICES=0 \
#    python3 ./models/research/object_detection/exporter_main_v2.py \
#            --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
#            --trained_checkpoint_dir=${CHECKPOINT_DIR} \
#            --output_directory=${OUTPUT_DIR} \
