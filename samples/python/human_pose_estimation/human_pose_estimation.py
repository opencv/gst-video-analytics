import sys
from argparse import ArgumentParser
import gi  # get Python bindings for GLib-based libraries
import math
import cv2
import numpy as np
from operator import itemgetter

gi.require_version('GstVideo', '1.0')
gi.require_version('Gst', '1.0')
gi.require_version('GObject', '2.0')
from gi.repository import Gst, GstVideo, GObject
# GVA API modules
from gstgva import VideoFrame, util

parser = ArgumentParser(add_help=False)
_args = parser.add_argument_group('Options')
_args.add_argument("-i", "--input", help="Required. Path to input video file",
                   required=True, type=str)
_args.add_argument("-d", "--detection_model", help="Required. Path to an .xml file with object detection model",
                   required=True, type=str)
_args.add_argument("-e", "--estimation_model", help="Required. Path to an .xml file with human-pose-estimation-0001",
                   required=True, type=str)
# init GStreamer
Gst.init(sys.argv)
limb_seq = [
    [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12], [12, 13],
    [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18], [3, 17], [6, 18]
]
map_idx = [
    [31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], [23, 24], [25, 26],
    [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], [55, 56], [37, 38], [45, 46]
]


def find_peaks(heatmap, all_peaks, prev_peak_counter):
    heatmap[heatmap < 0.1] = 0
    heatmap[np.isnan(heatmap)] = 0
    map_aug = np.zeros((heatmap.shape[0] + 2, heatmap.shape[1] + 2))
    map_left = np.zeros(map_aug.shape)
    map_right = np.zeros(map_aug.shape)
    map_up = np.zeros(map_aug.shape)
    map_down = np.zeros(map_aug.shape)

    map_aug[1:map_aug.shape[0] - 1, 1:map_aug.shape[1] - 1] = heatmap
    map_left[1:map_aug.shape[0] - 1, :map_aug.shape[1] - 2] = heatmap
    map_right[1:map_aug.shape[0] - 1, 2:map_aug.shape[1]] = heatmap
    map_up[:map_aug.shape[0] - 2, 1:map_aug.shape[1] - 1] = heatmap
    map_down[2:map_aug.shape[0], 1:map_aug.shape[1] - 1] = heatmap

    peaks_binary = (map_aug > map_left) & (map_aug > map_right) & (map_aug > map_up) & (map_aug > map_down)
    peaks_binary = peaks_binary[1:map_aug.shape[0] - 1, 1:map_aug.shape[1] - 1]
    peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))
    peaks = sorted(peaks, key=itemgetter(0))  # same order with matlab

    flag = np.ones(len(peaks), np.uint8)
    peaks_with_score_and_id = []
    peak_counter = 0
    for i, _ in enumerate(peaks):
        if flag[i] != 1:
            continue
        for j in range(i + 1, len(peaks)):
            if math.sqrt((peaks[i][0] - peaks[j][0]) ** 2 + (peaks[i][1] - peaks[j][1]) ** 2) < 6:
                flag[j] = 0
        peak_id = peak_counter + prev_peak_counter
        peak_counter += 1
        peaks_with_score_and_id.append([peaks[i][0], peaks[i][1], heatmap[peaks[i][1], peaks[i][0]], peak_id])
    all_peaks.append(peaks_with_score_and_id)
    return peak_counter


def _add_pose_single_candidate(subset, candidate, idx_joint, kpt_num=20):
    for joint in candidate:
        num = 0
        for subset_j in subset:  # check if already in some pose, was added as a part of another limb
            if subset_j[idx_joint] == joint[3]:
                num += 1
                continue
        if num == 0:
            person_keypoints = np.ones(kpt_num) * -1
            person_keypoints[idx_joint] = joint[3]  # joint idx
            person_keypoints[-1] = 1  # n joints in pose
            person_keypoints[-2] = joint[2]  # pose score
            subset.append(person_keypoints)

    return subset


def _filter_subset(subset):
    filtered_subset = []
    for subset_element in subset:
        if subset_element[-1] < 3 or (subset_element[-2] / subset_element[-1] < 0.2):
            continue
        filtered_subset.append(subset_element)

    return np.asarray(filtered_subset)


def _add_pose_both_candidates(subset, temp, index_a, index_b, candidates, kpt_num=20):
    for i, temp_i in enumerate(temp):
        num = 0
        for j, subset_j in enumerate(subset):
            if subset_j[index_a] == temp_i[0]:
                subset[j][index_b] = temp[i][1]
                num += 1
                subset[j][-1] += 1
                subset[j][-2] += candidates[temp_i[1], 2] + temp_i[2]
        if num == 0:
            person_keypoints = np.ones(kpt_num) * -1
            person_keypoints[index_a] = temp[i][0]
            person_keypoints[index_b] = temp[i][1]
            person_keypoints[-1] = 2
            person_keypoints[-2] = np.sum(candidates[temp_i[0:2], 2]) + temp_i[2]
            subset.append(person_keypoints)

    return subset


def _get_temperature(cand_a_, cand_b_, score_mid, pafs, threshold=0.05):
    temp_ = []
    for index_a_, cand_a_element in enumerate(cand_a_):
        for index_b_, cand_b_element in enumerate(cand_b_):
            mid_point = [(
                int(round((cand_a_element[0] + cand_b_element[0]) * 0.5)),
                int(round((cand_a_element[1] + cand_b_element[1]) * 0.5))
            )] * 2
            vec = [cand_b_element[0] - cand_a_element[0], cand_b_element[1] - cand_a_element[1]]
            norm_vec = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
            if norm_vec == 0:
                continue
            vec[0] /= norm_vec
            vec[1] /= norm_vec
            score_mid_a = score_mid[mid_point[0][1], mid_point[0][0], 0]
            score_mid_b = score_mid[mid_point[1][1], mid_point[1][0], 1]
            score = vec[0] * score_mid_a + vec[1] * score_mid_b

            height_n = pafs.shape[0] // 2
            suc_ratio = 0
            mid_score = 0
            mid_num = 10  # n points for integral over paf

            if score > -100:
                p_sum = 0
                p_count = 0

                x = np.linspace(cand_a_element[0], cand_b_element[0], mid_num)
                y = np.linspace(cand_a_element[1], cand_b_element[1], mid_num)
                for point_idx in range(0, mid_num):
                    px = int(round(x[point_idx]))
                    py = int(round(y[point_idx]))
                    pred = score_mid[py, px, 0:2]
                    score = vec[0] * pred[0] + vec[1] * pred[1]
                    if score > threshold:
                        p_sum += score
                        p_count += 1
                suc_ratio = p_count / mid_num
                ratio = 0
                if p_count > 0:
                    ratio = p_sum / p_count
                mid_score = ratio + min(height_n / norm_vec - 1, 0)
            if mid_score > 0 and suc_ratio > 0.8:
                score = mid_score
                score_all = score + cand_a_element[2] + cand_b_element[2]
                temp_.append([index_a_, index_b_, score, score_all])
    if temp_:
        temp_ = sorted(temp_, key=itemgetter(2), reverse=True)

    return temp_


def _copy_temperature_to_subset(subset, temp, index_a, index_b):
    for _, temp_i in enumerate(temp):
        for j, subset_j in enumerate(subset):
            check_subset_a = subset_j[index_a] == temp_i[0] and subset_j[index_b] == -1
            check_subset_b = subset_j[index_b] == temp_i[1] and subset_j[index_a] == -1
            if check_subset_a:
                subset[j][index_b] = temp_i[1]
                continue
            if check_subset_b:
                subset[j][index_a] = temp_i[0]

    return subset


def _get_connections(cand_a, cand_b, score_mid, pafs, thresh):
    temp_ = _get_temperature(cand_a, cand_b, score_mid, pafs, thresh)
    num_limbs = min(len(cand_a), len(cand_b))
    cnt = 0
    occur_a = np.zeros(len(cand_a), dtype=np.int32)
    occur_b = np.zeros(len(cand_b), dtype=np.int32)
    connections = []
    for row_temp in temp_:
        if cnt == num_limbs:
            break
        i, j, score = row_temp[0:3]
        if occur_a[i] == 0 and occur_b[j] == 0:
            connections.append([cand_a[i][3], cand_b[j][3], score])
            cnt += 1
            occur_a[i] = 1
            occur_b[j] = 1
    return connections


def group_peaks(peaks, pafs, kpt_num=20, threshold=0.05):
    subset = []
    candidates = np.array([item for sublist in peaks for item in sublist])
    for keypoint_id, maped_keypoints in enumerate(map_idx):
        score_mid = pafs[:, :, [x - 19 for x in maped_keypoints]]
        candidate_a = peaks[limb_seq[keypoint_id][0] - 1]
        candidate_b = peaks[limb_seq[keypoint_id][1] - 1]
        idx_joint_a = limb_seq[keypoint_id][0] - 1
        idx_joint_b = limb_seq[keypoint_id][1] - 1

        if not candidate_a and not candidate_b:  # no such limb
            continue
        if not candidate_a:  # limb has just B joint
            subset = _add_pose_single_candidate(subset, candidate_b, idx_joint_b, kpt_num)
            continue
        if not candidate_b:  # limb has just A joint
            subset = _add_pose_single_candidate(subset, candidate_a, idx_joint_a, kpt_num)
            continue

        temp = _get_connections(candidate_a, candidate_b, score_mid, pafs, threshold)
        if not temp:
            continue

        if keypoint_id == 0:
            subset = [np.ones(kpt_num) * -1 for _ in temp]
            for i, temp_i in enumerate(temp):
                subset[i][limb_seq[0][0] - 1] = temp_i[0]
                subset[i][limb_seq[0][1] - 1] = temp_i[1]
                subset[i][-1] = 2
                subset[i][-2] = np.sum(candidates[temp_i[0:2], 2]) + temp_i[2]
        else:
            index_a = limb_seq[keypoint_id][0] - 1
            index_b = limb_seq[keypoint_id][1] - 1
            if keypoint_id in (17, 18):
                subset = _copy_temperature_to_subset(subset, temp, index_a, index_b)
                continue
            subset = _add_pose_both_candidates(subset, temp, index_a, index_b, candidates, kpt_num)

    return _filter_subset(subset), candidates


def get_poses(subset, candidate):
    persons_keypoints_x, persons_keypoints_y, persons_keypoints_v = [], [], []
    scores = []
    for subset_element in subset:
        if subset_element.size == 0:
            continue
        keypoints_x, keypoints_y, keypoints_v = [0] * 17, [0] * 17, [0] * 17
        to_coco_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        person_score = subset_element[-2]
        position_id = -1
        for keypoint_id in subset_element[:-2]:
            position_id += 1
            if position_id == 1:  # No 'Neck' in COCO
                continue

            cx, cy, visibility = 0, 0, 0  # Keypoint not found
            if keypoint_id != -1:
                cx, cy = candidate[keypoint_id.astype(int), 0:2]
                cx = cx - 0.5 + 1  # +1 for matlab consistency, coords start from 1
                cy = cy - 0.5 + 1
                visibility = 1
            keypoints_x[to_coco_map[position_id]] = cx
            keypoints_y[to_coco_map[position_id]] = cy
            keypoints_v[to_coco_map[position_id]] = visibility

        scores.append(person_score * max(0, (subset_element[-1] - 1)))  # -1 for Neck
        persons_keypoints_x.append(keypoints_x)
        persons_keypoints_y.append(keypoints_y)
        persons_keypoints_v.append(keypoints_v)

    persons_keypoints_x = np.array(persons_keypoints_x)
    persons_keypoints_y = np.array(persons_keypoints_y)
    persons_keypoints_v = np.array(persons_keypoints_v)
    scores = np.array(scores)

    return persons_keypoints_x, persons_keypoints_y, persons_keypoints_v, scores


def process(frame):
    paf = list(frame.tensors())[0].data()
    paf = np.reshape(paf, list(frame.tensors())[0].dims())
    heatmap = list(frame.tensors())[1].data()
    heatmap = np.reshape(heatmap, list(frame.tensors())[1].dims())
    height = frame.video_info().height
    width = frame.video_info().width
    heatmap_avg = np.zeros((height, width, 19), dtype=np.float32)
    paf_avg = np.zeros((height, width, 38), dtype=np.float32)
    pad = [0, 0, 0, 0]
    transpose_order = (1, 2, 0)

    heatmap = np.transpose(np.squeeze(heatmap), transpose_order)

    heatmap = cv2.resize(heatmap, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    heatmap = heatmap[pad[0]:heatmap.shape[0] - pad[2], pad[1]:heatmap.shape[1] - pad[3]:, :]
    heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
    heatmap_avg = heatmap_avg + heatmap

    paf = np.transpose(np.squeeze(paf), transpose_order)
    paf = cv2.resize(paf, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    paf = paf[pad[0]:paf.shape[0] - pad[2], pad[1]:paf.shape[1] - pad[3], :]
    paf = cv2.resize(paf, (width, height), interpolation=cv2.INTER_CUBIC)
    paf_avg = paf_avg + paf

    peak_counter = 0
    all_peaks = []
    for part in range(0, 18):  # 19th for bg
        peak_counter += find_peaks(heatmap_avg[:, :, part], all_peaks, peak_counter)

    subset, candidate = group_peaks(all_peaks, paf_avg)

    poses = get_poses(subset, candidate)
    return poses


mapping = [[6, 8], [5, 7], [8, 10], [7, 9], [12, 14], [11, 13]]


def frame_callback(frame: VideoFrame):
    color = (255, 0, 0)
    with frame.data() as mat:
        poses = process(frame)
        x_values = poses[0]
        y_values = poses[1]
        for person in range(len(x_values)):
            for i in range(len(x_values[0])):
                cv2.circle(mat, (int(x_values[person][i]), int(y_values[person][i])), 3, color, -1)
            for j in range(len(mapping)):
                start_idx = mapping[j][0]
                end_idx = mapping[j][1]
                cv2.line(mat, (int(x_values[person][start_idx]), int(y_values[person][start_idx])),
                         (int(x_values[person][end_idx]), int(y_values[person][end_idx])), color, 3)

    return True


def pad_probe_callback(pad, info):
    with util.GST_PAD_PROBE_INFO_BUFFER(info) as buffer:
        caps = pad.get_current_caps()
        frame = VideoFrame(buffer, caps=caps)
        status = frame_callback(frame)
    return Gst.PadProbeReturn.OK if status else Gst.PadProbeReturn.DROP


def set_callback(pipeline):
    gvawatermark = pipeline.get_by_name("gvawatermark")
    if gvawatermark:
        pad = gvawatermark.get_static_pad('src')
        pad.add_probe(Gst.PadProbeType.BUFFER, pad_probe_callback)


def create_launch_string(args):
    if "/dev/video" in args.input:
        source = "v4l2src device"
    elif "://" in args.input:
        source = "urisourcebin buffer-size=4096 uri"
    else:
        source = "filesrc location"

    return "{}={} ! decodebin ! \
    videoconvert n-threads=4 ! capsfilter caps=\"video/x-raw,format=BGRx\" ! \
    gvadetect model={} device=CPU batch-size=1 ! queue ! \
    gvainference name=gvainference model={} ! queue ! \
    gvawatermark name=gvawatermark ! videoconvert n-threads=4 ! \
    fpsdisplaysink video-sink=xvimagesink sync=false".format(source, args.input, args.detection_model,
                                                             args.estimation_model)


def main():
    Gst.init(sys.argv)
    args = parser.parse_args()
    # build pipeline using parse_launch
    pipeline_str = create_launch_string(args)
    pipeline = Gst.parse_launch(pipeline_str)
    # set callback
    set_callback(pipeline)

    # start pipeline
    pipeline.set_state(Gst.State.PLAYING)
    # wait until EOS or error
    bus = pipeline.get_bus()
    msg = bus.timed_pop_filtered(
        Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS)
    # free pipeline
    pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    sys.exit(main() or 0)
