/*******************************************************************************
 * Copyright (C) 2020-2021 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "converters/yolo_v3_base.h"

#include "inference_backend/logger.h"

#include <cmath>
#include <sstream>

using namespace DetectionPlugin;
using namespace Converters;

YOLOV3Converter::YOLOV3Converter(size_t classes_number, std::vector<float> anchors,
                                 std::map<size_t, std::vector<size_t>> masks, size_t cells_number_x,
                                 size_t cells_number_y, double iou_threshold, size_t bbox_number_on_cell,
                                 size_t input_size_h, size_t input_size_w, bool do_cls_softmax, bool output_sigmoid_activation)
    : YOLOConverter(anchors, iou_threshold, {classes_number, cells_number_x, cells_number_y, bbox_number_on_cell},
                    do_cls_softmax, output_sigmoid_activation),
      masks(masks), input_size_h(input_size_h), input_size_w(input_size_w) {
}

std::vector<float> YOLOV3Converter::softmax(const float *arr, size_t side_h, size_t side_w, size_t common_offset, size_t size) {
    std::vector<float> sftm_arr(size);
    float sum = 0;
    for (size_t i = 0; i < size; ++i) {
        const size_t class_index = entryIndex(side_h, side_w, common_offset, 5 + i);
        sftm_arr[i] = std::exp(arr[class_index]);
        sum += sftm_arr[i];
    }
    for (size_t i = 0; i < size; ++i) {
        sftm_arr[i] /= sum;
    }
    return sftm_arr;
}

bool YOLOV3Converter::process(const std::map<std::string, InferenceBackend::OutputBlob::Ptr> &output_blobs,
                              const std::vector<std::shared_ptr<InferenceFrame>> &frames,
                              GstStructure *detection_result, double confidence_threshold, GValueArray *labels) {
    bool flag = false;
    try {
        if (frames.size() != 1) {
            std::string err = "Batch size other than 1 is not supported";
            const gchar *converter = gst_structure_get_string(detection_result, "converter");
            if (converter)
                err += " for this post processor: " + std::string(converter);
            throw std::invalid_argument(err);
        }

        std::vector<DetectedObject> objects;
        for (const auto &blob_iter : output_blobs) {
            InferenceBackend::OutputBlob::Ptr blob = blob_iter.second;
            if (not blob)
                throw std::invalid_argument("Output blob is nullptr");

            auto dims = blob->GetDims();
            if (dims.size() != 4) {
                throw std::runtime_error("Invalid output blob dimensions");
            }

            const size_t side_h = dims[2];
            const size_t side_w = dims[3];
            size_t anchor_offset = 0;
            if (masks.find(side_h) == masks.cend()) {
                std::ostringstream msg_stream;
                msg_stream << "Mismatch between the size of the bounding box in the mask: " << masks.cbegin()->first
                           << " - and the actual size of the bounding "
                              "box: "
                           << side_h << "x" << side_w << ".";

                throw std::runtime_error(msg_stream.str());
            }

            /* we must check blob size only for layers which size we specify */
            if (side_h == output_shape_info.cells_number_x) {
                size_t blob_size = 1;
                for (const auto &dim : dims)
                    blob_size *= dim;
                if (blob_size != output_shape_info.requied_blob_size)
                    throw std::runtime_error("Size of the resulting output blob (" + std::to_string(blob_size) +
                                             ") does not match the required (" +
                                             std::to_string(output_shape_info.requied_blob_size) + ").");
            }
            
            std::vector<size_t> mask = masks.at(side_h);
            anchor_offset = 2 * mask[0];
            const float *blob_data = (const float *)blob->GetData();
            if (not blob_data)
                throw std::invalid_argument("Output blob data is nullptr");

            const size_t side_square = side_h * side_w;
            for (size_t i = 0; i < side_square; ++i) {
                const size_t row = i / side_w;
                const size_t col = i % side_w;
                for (size_t bbox_cell_num = 0; bbox_cell_num < output_shape_info.bbox_number_on_cell; ++bbox_cell_num) {
                    const size_t common_offset = bbox_cell_num * side_square + i;
                    const size_t bbox_conf_index = entryIndex(side_h, side_w, common_offset, coords);
                    const size_t bbox_index = entryIndex(side_h, side_w, common_offset, 0);

                    float bbox_conf = blob_data[bbox_conf_index];
                    if (output_sigmoid_activation)
                        bbox_conf = sigmoid(bbox_conf);
                    if (bbox_conf < confidence_threshold)
                        continue;

                    std::pair<size_t, float> bbox_class = std::make_pair(0, 0.f);
                    if (do_cls_softmax) {
                        const auto cls_confs =
                            softmax(blob_data, side_h, side_w, common_offset, output_shape_info.classes_number);

                        for (size_t bbox_class_id = 0; bbox_class_id < output_shape_info.classes_number;
                             ++bbox_class_id) {
                            const float bbox_class_prob = cls_confs[bbox_class_id];

                            if (bbox_class_prob > 1.f && bbox_class_prob < 0.f) {
                                GST_WARNING("bbox_class_prob is weird %f", bbox_class_prob);
                            }
                            if (bbox_class_prob > bbox_class.second) {
                                bbox_class.first = bbox_class_id;
                                bbox_class.second = bbox_class_prob;
                            }
                        }
                    } else {
                        for (size_t bbox_class_id = 0; bbox_class_id < output_shape_info.classes_number;
                             ++bbox_class_id) {
                            const size_t class_index = entryIndex(side_h, side_w, common_offset, 5 + bbox_class_id);
                            const float bbox_class_prob = blob_data[class_index];

                            if (bbox_class_prob > 1.f && bbox_class_prob < 0.f) {
                                GST_WARNING("bbox_class_prob is weird %f", bbox_class_prob);
                            }
                            if (bbox_class_prob > bbox_class.second) {
                                bbox_class.first = bbox_class_id;
                                bbox_class.second = bbox_class_prob;
                            }
                        }
                    }

                    const float confidence = bbox_conf * bbox_class.second;
                    if (confidence < confidence_threshold)
                        continue;

                    // TODO: check if index in array range
                    const float raw_x = blob_data[bbox_index + 0 * side_square];
                    const float raw_y = blob_data[bbox_index + 1 * side_square];
                    const float raw_w = blob_data[bbox_index + 2 * side_square];
                    const float raw_h = blob_data[bbox_index + 3 * side_square];

                    const float x = static_cast<float>(col + (output_sigmoid_activation ? sigmoid(raw_x) : raw_x)) /
                                    side_w * input_size_w;
                    const float y = static_cast<float>(row + (output_sigmoid_activation ? sigmoid(raw_y) : raw_y)) /
                                    side_h * input_size_h;

                    // TODO: check if index in array range
                    const float width = std::exp(raw_w) * anchors[anchor_offset + 2 * bbox_cell_num];
                    const float height = std::exp(raw_h) * anchors[anchor_offset + 2 * bbox_cell_num + 1];

                    DetectedObject obj(x, y, width, height, bbox_class.first, confidence, 1.0f / input_size_h,
                                       1.0f / input_size_w);
                    objects.push_back(obj);
                }
            }
        }
        storeObjects(objects, frames[0], detection_result, labels);
        flag = true;
    } catch (const std::exception &e) {
        std::throw_with_nested(std::runtime_error("Failed to do YoloV3 post-processing"));
    }
    return flag;
}

size_t YOLOV3Converter::entryIndex(size_t side_h, size_t side_w, size_t location, size_t entry) {
    size_t side_square = side_h * side_w;
    size_t bbox_cell_num = location / side_square;
    size_t loc = location % side_square;
    // side_square is the tensor dimension of the YoloV3 model. Overflow is not possible here.
    return side_square * (bbox_cell_num * (output_shape_info.classes_number + 5) + entry) + loc;
}
