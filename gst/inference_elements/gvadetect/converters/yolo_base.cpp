/*******************************************************************************
 * Copyright (C) 2020-2021 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "converters/yolo_base.h"
#include "converters/yolo_v2_base.h"
#include "converters/yolo_v3_base.h"

#include "inference_backend/logger.h"

#include <algorithm>
#include <mutex>

using namespace DetectionPlugin;
using namespace Converters;

size_t getClassesNum(const GstStructure *s);
size_t getCellsNumberX(const GstStructure *s);
size_t getCellsNumberY(const GstStructure *s);
double getIOUThreshold(const GstStructure *s);
size_t getBboxNumberOnCell(const GstStructure *s);
bool getDoClsSoftmax(const GstStructure *s);
bool getOutputSigmoidActivation(const GstStructure *s);
std::vector<float> getAnchors(const GstStructure *s);
std::map<size_t, std::vector<size_t>> getMask(const GstStructure *s, size_t bbox_number_on_cell, size_t cells_number);

YOLOConverter *YOLOConverter::makeYOLOConverter(const std::string &converter_type,
                                                const GstStructure *output_model_proc_info,
                                                const ModelInputInfo &input_info) {
    const auto classes_number = getClassesNum(output_model_proc_info);
    const auto anchors = getAnchors(output_model_proc_info);
    const auto iou_threshold = getIOUThreshold(output_model_proc_info);
    const auto cells_number_x = getCellsNumberX(output_model_proc_info);
    const auto cells_number_y = getCellsNumberY(output_model_proc_info);
    const auto do_cls_softmax = getDoClsSoftmax(output_model_proc_info);
    const auto output_sigmoid_activation = getOutputSigmoidActivation(output_model_proc_info);
    auto bbox_number_on_cell = getBboxNumberOnCell(output_model_proc_info);

    if (converter_type == "tensor_to_bbox_yolo_v2") {
        if (!bbox_number_on_cell)
            bbox_number_on_cell = 5;
        return new YOLOV2Converter(classes_number, anchors, cells_number_x, cells_number_y, iou_threshold,
                                   bbox_number_on_cell, do_cls_softmax, output_sigmoid_activation);
    }
    if (converter_type == "tensor_to_bbox_yolo_v3") {
        if (!bbox_number_on_cell)
            bbox_number_on_cell = 3;
        const auto masks = getMask(output_model_proc_info, bbox_number_on_cell, cells_number_y);
        if (input_info.width / 32 != cells_number_x || input_info.height / 32 != cells_number_y) {
            GST_WARNING("The size of the input layer of the model does not match the specified number of cells. Verify "
                        "your \"cells_number_x\" and \"cells_number_y\" field in model_proc.");
        }
        return new YOLOV3Converter(classes_number, anchors, masks, cells_number_x, cells_number_y, iou_threshold,
                                   bbox_number_on_cell, input_info.height, input_info.width, do_cls_softmax, output_sigmoid_activation);
    }
    return nullptr;
}

void YOLOConverter::storeObjects(std::vector<DetectedObject> &objects, const std::shared_ptr<InferenceFrame> frame,
                                 GstStructure *detection_result, GValueArray *labels) {
    ITT_TASK(__FUNCTION__);
    runNms(objects);

    for (DetectedObject &object : objects) {
        addRoi(frame, object.x, object.y, object.w, object.h, object.class_id, object.confidence,
               gst_structure_copy(detection_result),
               labels); // each ROI gets its own copy, which is then
                        // owned by GstVideoRegionOfInterestMeta
    }
}

void YOLOConverter::runNms(std::vector<DetectedObject> &candidates) {
    ITT_TASK(__FUNCTION__);
    std::sort(candidates.rbegin(), candidates.rend());

    for (auto p_first_candidate = candidates.begin(); p_first_candidate != candidates.end(); ++p_first_candidate) {
        const auto &first_candidate = *p_first_candidate;
        double first_candidate_area = first_candidate.w * first_candidate.h;

        for (auto p_candidate = p_first_candidate + 1; p_candidate != candidates.end();) {
            const auto &candidate = *p_candidate;

            gdouble inter_width = std::min(first_candidate.x + first_candidate.w, candidate.x + candidate.w) -
                                  std::max(first_candidate.x, candidate.x);
            gdouble inter_height = std::min(first_candidate.y + first_candidate.h, candidate.y + candidate.h) -
                                   std::max(first_candidate.y, candidate.y);
            if (inter_width <= 0.0 || inter_height <= 0.0) {
                ++p_candidate;
                continue;
            }

            gdouble inter_area = inter_width * inter_height;
            gdouble candidate_area = candidate.w * candidate.h;

            gdouble overlap = inter_area / (candidate_area + first_candidate_area - inter_area);
            if (overlap > iou_threshold)
                p_candidate = candidates.erase(p_candidate);
            else
                ++p_candidate;
        }
    }
}

std::vector<float> getAnchors(const GstStructure *s) {
    if (!gst_structure_has_field(s, "anchors"))
        throw std::runtime_error("model proc does not have \"anchors\" parameter.");
    GValueArray *arr = NULL;
    gst_structure_get_array(const_cast<GstStructure *>(s), "anchors", &arr);
    std::vector<float> anchors;
    if (arr) {
        anchors.reserve(arr->n_values);
        for (guint i = 0; i < arr->n_values; ++i)
            anchors.push_back(g_value_get_double(g_value_array_get_nth(arr, i)));
        g_value_array_free(arr);
    } else {
        throw std::runtime_error("\"anchors\" array is null.");
    }
    return anchors;
}

std::map<size_t, std::vector<size_t>> getMask(const GstStructure *s, size_t bbox_number_on_cell, size_t cells_number_y) {

    if (!gst_structure_has_field(s, "masks"))
        throw std::runtime_error("model proc does not have \"masks\" parameter.");
    GValueArray *arr = NULL;
    size_t side = cells_number_y;
    gst_structure_get_array(const_cast<GstStructure *>(s), "masks", &arr);
    std::vector<size_t> masks;
    if (arr) {
        masks.reserve(arr->n_values);
        for (guint i = 0; i < arr->n_values; ++i)
            masks.push_back(g_value_get_int(g_value_array_get_nth(arr, i)));
        g_value_array_free(arr);
    } else {
        throw std::runtime_error("\"masks\" array is null.");
    }
    std::map<size_t, std::vector<size_t>> mask;
    std::vector<size_t> one_side_mask;
    one_side_mask.reserve(bbox_number_on_cell);
    for (size_t i = 0; i < masks.size(); ++i) {
        if (i != 0 && i % bbox_number_on_cell == 0) {
            mask.insert({side, one_side_mask});
            side *= 2;
            one_side_mask.clear();
        }
        one_side_mask.push_back(masks[i]);
    }
    mask.insert({side, one_side_mask});
    return mask;
}

size_t getClassesNum(const GstStructure *s) {
    int classes = 0;
    if (gst_structure_has_field(s, "classes")) {
        gst_structure_get_int(s, "classes", &classes);
    } else {
        throw std::runtime_error("model proc does not have \"classes\" parameter.");
    }
    return classes;
}

size_t getCellsNumberX(const GstStructure *s) {
    int cells_number_x = 13;
    if (gst_structure_has_field(s, "cells_number_x")) {
        gst_structure_get_int(s, "cells_number_x", &cells_number_x);
    }

    return cells_number_x;
}

size_t getCellsNumberY(const GstStructure *s) {
    int cells_number_y = 13;
    if (gst_structure_has_field(s, "cells_number_y")) {
        gst_structure_get_int(s, "cells_number_y", &cells_number_y);
    }

    return cells_number_y;
}

size_t getBboxNumberOnCell(const GstStructure *s) {
    int bbox_number_on_cell = 0;
    if (gst_structure_has_field(s, "bbox_number_on_cell")) {
        gst_structure_get_int(s, "bbox_number_on_cell", &bbox_number_on_cell);
    } else {
        GST_WARNING("model proc does not have \"bbox_number_on_cell\" parameter.");
    }
    return bbox_number_on_cell;
}

double getIOUThreshold(const GstStructure *s) {
    double iou_threshold = 0.5;
    if (gst_structure_has_field(s, "iou_threshold")) {
        gst_structure_get_double(s, "iou_threshold", &iou_threshold);
    }

    return iou_threshold;
}

bool getDoClsSoftmax(const GstStructure *s) {
    gboolean do_cls_sftm = FALSE;
    if (gst_structure_has_field(s, "do_cls_softmax")) {
        gst_structure_get_boolean(s, "do_cls_softmax", &do_cls_sftm);
    }

    return do_cls_sftm;
}

bool getOutputSigmoidActivation(const GstStructure *s) {
    gboolean do_coords_sgmd = FALSE;
    if (gst_structure_has_field(s, "output_sigmoid_activation")) {
        gst_structure_get_boolean(s, "output_sigmoid_activation", &do_coords_sgmd);
    }

    return do_coords_sgmd;
}
