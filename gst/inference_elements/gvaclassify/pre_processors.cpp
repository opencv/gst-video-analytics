/*******************************************************************************
 * Copyright (C) 2018-2020 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include <map>
#include <string>
#include <vector>

#include "classification_history.h"
#include "gstgvaclassify.h"
#include "gva_base_inference.h"
#include "inference_backend/safe_arithmetic.h"
#include "region_of_interest.h"
#include "utils.h"

#include <inference_backend/image_inference.h>
#include <opencv2/imgproc.hpp>

#include "pre_processors.h"

namespace {

using namespace InferenceBackend;

bool IsROIClassificationNeeded(GvaBaseInference *gva_base_inference, guint64 current_num_frame,
                               GstBuffer *buffer, GstVideoRegionOfInterestMeta *roi) {
    GstGvaClassify *gva_classify = (GstGvaClassify *)gva_base_inference;

    // Check is object-class same with roi type
    if (gva_classify->object_class[0]) {
        static std::map<std::string, std::vector<std::string>> elemets_object_classes;
        auto it = elemets_object_classes.find(gva_base_inference->model_instance_id);
        if (it == elemets_object_classes.end())
            it = elemets_object_classes.insert(
                it, {gva_base_inference->model_instance_id, Utils::splitString(gva_classify->object_class, ',')});

        auto compare_quark_string = [roi](const std::string &str) {
            const gchar *roi_type = roi->roi_type ? g_quark_to_string(roi->roi_type) : "";
            return (strcmp(roi_type, str.c_str()) == 0);
        };
        if (std::find_if(it->second.cbegin(), it->second.cend(), compare_quark_string) == it->second.cend()) {
            return false;
        }
    }

    assert(gva_classify->classification_history != NULL);
    // If reclassification is possible on every interval, check the metadata to see
    // if classification of this ROI should be skipped.
    if (gva_classify->reclassify_interval == 1) {
        return gva_classify->classification_history->IsROIClassificationNeededDueToMeta(buffer, roi);
    }

    // Check is object recently classified
    return (gva_classify->classification_history->IsROIClassificationNeeded(buffer, roi, current_num_frame));
}

} // anonymous namespace

IsROIClassificationNeededFunction IS_ROI_CLASSIFICATION_NEEDED = IsROIClassificationNeeded;
