#!/bin/bash
# ==============================================================================
# Copyright (C) 2018-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT
# ==============================================================================

set -e

INPUT=${1:-https://github.com/intel-iot-devkit/safety-gear-detector-python/raw/master/resources/Safety_Full_Hat_and_Vest.mp4}

GET_MODEL_PATH() {
    model_name=$1
    precision=${2:-"FP32"}
    for models_dir in ${MODELS_PATH//:/ }; do
        paths=$(find $models_dir -type f -name "*$model_name.xml" -print)
        if [ ! -z "$paths" ];
        then
            considered_precision_paths=$(echo "$paths" | grep "/$precision/")
           if [ ! -z "$considered_precision_paths" ];
            then
                echo $(echo "$considered_precision_paths" | head -n 1)
                exit 0
            else
                echo $(echo "$paths" | head -n 1)
                exit 0
            fi
        fi
    done

    echo -e "\e[31mModel $model_name file was not found. Please set MODELS_PATH\e[0m" 1>&2
    exit 1
}

PROC_PATH() {
    echo ./model_proc/$1.json
}

MODEL_D=person-vehicle-bike-detection-crossroad-0078
MODEL_E=human-pose-estimation-0001

PATH_D=$(GET_MODEL_PATH $MODEL_D)
PATH_E=$(GET_MODEL_PATH $MODEL_E)

echo Running sample with the following parameters:
echo GST_PLUGIN_PATH=${GST_PLUGIN_PATH}

PYTHONPATH=$PYTHONPATH:$(dirname "$0")/../../../python \
python3 $(dirname "$0")/human_pose_estimation.py -i ${INPUT} -d ${PATH_D} -e ${PATH_E}
