#/usr/bin/bash

SAVED_MODEL_DIR=notebook/retinanet
OUT_TFLITE_PATH=${SAVED_MODEL_DIR}/saved_model.tflite

python3 convert_tflite.py \
        --saved_model_dir=${SAVED_MODEL_DIR} \
        --out_tflite_path=${OUT_TFLITE_PATH}
