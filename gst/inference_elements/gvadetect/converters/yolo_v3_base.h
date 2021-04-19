/*******************************************************************************
 * Copyright (C) 2020-2021 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/
#pragma once

#include "converters/yolo_base.h"

namespace DetectionPlugin {
namespace Converters {

class YOLOV3Converter : public YOLOConverter {
  protected:
    const std::map<size_t, std::vector<size_t>> masks;
    const size_t coords = 4;
    const size_t input_size_h;
    const size_t input_size_w;

    size_t entryIndex(size_t side_h, size_t side_w,  size_t location, size_t entry);
    std::vector<float> softmax(const float *arr, size_t side_h, size_t side_w, size_t common_offset, size_t size);

  public:
    YOLOV3Converter(size_t classes_number, std::vector<float> anchors, std::map<size_t, std::vector<size_t>> masks,
                    size_t cells_number_x, size_t cells_number_y, double iou_threshold = 0.5,
                    size_t bbox_number_on_cell = 3, size_t input_size_h = 416, size_t input_size_w = 416, bool do_cls_softmax = false,
                    bool output_sigmoid_activation = false);

    bool process(const std::map<std::string, InferenceBackend::OutputBlob::Ptr> &output_blobs,
                 const std::vector<std::shared_ptr<InferenceFrame>> &frames, GstStructure *detection_result,
                 double confidence_threshold, GValueArray *labels);
};

} // namespace Converters
} // namespace DetectionPlugin
