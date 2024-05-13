# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))

os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import cv2
import numpy as np
import time
import sys

# import tools.infer.utility as utility
import utility
from utility import get_logger

from DBPostProcess import DBPostProcess
from operators import DetResizeForTest, KeepKeys, NormalizeImage, ToCHWImage


def transform(data, ops=None):
    """transform"""
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data

logger = get_logger()


class TextDetector(object):
    def __init__(self, args):
        self.args = args
        self.det_algorithm = args.det_algorithm
        self.use_onnx = args.use_onnx
        postprocess_params = {}
        assert self.det_algorithm == "DB"
        postprocess_params["name"] = "DBPostProcess"
        postprocess_params["thresh"] = args.det_db_thresh
        postprocess_params["box_thresh"] = args.det_db_box_thresh
        postprocess_params["max_candidates"] = 1000
        postprocess_params["unclip_ratio"] = args.det_db_unclip_ratio
        postprocess_params["use_dilation"] = args.use_dilation
        postprocess_params["score_mode"] = args.det_db_score_mode
        postprocess_params["box_type"] = args.det_box_type

        self.preprocess_op = [
            DetResizeForTest(limit_side_len=args.det_limit_side_len, limit_type=args.det_limit_type),
            NormalizeImage(std= [0.229, 0.224, 0.225], mean= [0.485, 0.456, 0.406], scale= 1./255., order= "hwc"),
            ToCHWImage(),
            KeepKeys(keep_keys= ["image", "shape"])
        ]
        self.postprocess_op = DBPostProcess(**postprocess_params)
        (
            self.predictor,
            self.input_tensor,
            self.output_tensors,
            self.config,
        ) = utility.create_predictor(args, "det", logger)

        assert self.use_onnx
        if self.use_onnx:
            img_h, img_w = self.input_tensor.shape[2:]
            if isinstance(img_h, str) or isinstance(img_w, str):
                pass
            elif img_h is not None and img_w is not None and img_h > 0 and img_w > 0:
                self.preprocess_op[0] = DetResizeForTest(image_shape=[img_h, img_w])


    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def predict(self, img):
        ori_im = img.copy()
        data = {"image": img}

        st = time.time()

        if self.args.benchmark:
            self.autolog.times.start()

        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()

        if self.args.benchmark:
            self.autolog.times.stamp()
        if self.use_onnx:
            input_dict = {}
            input_dict[self.input_tensor.name] = img
            outputs = self.predictor.run(self.output_tensors, input_dict)
        else:
            self.input_tensor.copy_from_cpu(img)
            self.predictor.run()
            outputs = []
            for output_tensor in self.output_tensors:
                output = output_tensor.copy_to_cpu()
                outputs.append(output)
            if self.args.benchmark:
                self.autolog.times.stamp()

        preds = {}
        if self.det_algorithm == "EAST":
            preds["f_geo"] = outputs[0]
            preds["f_score"] = outputs[1]
        elif self.det_algorithm == "SAST":
            preds["f_border"] = outputs[0]
            preds["f_score"] = outputs[1]
            preds["f_tco"] = outputs[2]
            preds["f_tvo"] = outputs[3]
        elif self.det_algorithm in ["DB", "PSE", "DB++"]:
            preds["maps"] = outputs[0]
        elif self.det_algorithm == "FCE":
            for i, output in enumerate(outputs):
                preds["level_{}".format(i)] = output
        elif self.det_algorithm == "CT":
            preds["maps"] = outputs[0]
            preds["score"] = outputs[1]
        else:
            raise NotImplementedError

        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]["points"]

        if self.args.det_box_type == "poly":
            dt_boxes = self.filter_tag_det_res_only_clip(dt_boxes, ori_im.shape)
        else:
            dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)

        if self.args.benchmark:
            self.autolog.times.end(stamp=True)
        et = time.time()
        return dt_boxes, et - st

    def __call__(self, img):
        # For image like poster with one side much greater than the other side,
        # splitting recursively and processing with overlap to enhance performance.
        MIN_BOUND_DISTANCE = 50
        dt_boxes = np.zeros((0, 4, 2), dtype=np.float32)
        elapse = 0
        if (
            img.shape[0] / img.shape[1] > 2
            and img.shape[0] > self.args.det_limit_side_len
        ):
            start_h = 0
            end_h = 0
            while end_h <= img.shape[0]:
                end_h = start_h + img.shape[1] * 3 // 4
                subimg = img[start_h:end_h, :]
                if len(subimg) == 0:
                    break
                sub_dt_boxes, sub_elapse = self.predict(subimg)
                offset = start_h
                # To prevent text blocks from being cut off, roll back a certain buffer area.
                if (
                    len(sub_dt_boxes) == 0
                    or img.shape[1] - max([x[-1][1] for x in sub_dt_boxes])
                    > MIN_BOUND_DISTANCE
                ):
                    start_h = end_h
                else:
                    sorted_indices = np.argsort(sub_dt_boxes[:, 2, 1])
                    sub_dt_boxes = sub_dt_boxes[sorted_indices]
                    bottom_line = (
                        0
                        if len(sub_dt_boxes) <= 1
                        else int(np.max(sub_dt_boxes[:-1, 2, 1]))
                    )
                    if bottom_line > 0:
                        start_h += bottom_line
                        sub_dt_boxes = sub_dt_boxes[
                            sub_dt_boxes[:, 2, 1] <= bottom_line
                        ]
                    else:
                        start_h = end_h
                if len(sub_dt_boxes) > 0:
                    if dt_boxes.shape[0] == 0:
                        dt_boxes = sub_dt_boxes + np.array(
                            [0, offset], dtype=np.float32
                        )
                    else:
                        dt_boxes = np.append(
                            dt_boxes,
                            sub_dt_boxes + np.array([0, offset], dtype=np.float32),
                            axis=0,
                        )
                elapse += sub_elapse
        elif (
            img.shape[1] / img.shape[0] > 3
            and img.shape[1] > self.args.det_limit_side_len * 3
        ):
            start_w = 0
            end_w = 0
            while end_w <= img.shape[1]:
                end_w = start_w + img.shape[0] * 3 // 4
                subimg = img[:, start_w:end_w]
                if len(subimg) == 0:
                    break
                sub_dt_boxes, sub_elapse = self.predict(subimg)
                offset = start_w
                if (
                    len(sub_dt_boxes) == 0
                    or img.shape[0] - max([x[-1][0] for x in sub_dt_boxes])
                    > MIN_BOUND_DISTANCE
                ):
                    start_w = end_w
                else:
                    sorted_indices = np.argsort(sub_dt_boxes[:, 2, 0])
                    sub_dt_boxes = sub_dt_boxes[sorted_indices]
                    right_line = (
                        0
                        if len(sub_dt_boxes) <= 1
                        else int(np.max(sub_dt_boxes[:-1, 1, 0]))
                    )
                    if right_line > 0:
                        start_w += right_line
                        sub_dt_boxes = sub_dt_boxes[sub_dt_boxes[:, 1, 0] <= right_line]
                    else:
                        start_w = end_w
                if len(sub_dt_boxes) > 0:
                    if dt_boxes.shape[0] == 0:
                        dt_boxes = sub_dt_boxes + np.array(
                            [offset, 0], dtype=np.float32
                        )
                    else:
                        dt_boxes = np.append(
                            dt_boxes,
                            sub_dt_boxes + np.array([offset, 0], dtype=np.float32),
                            axis=0,
                        )
                elapse += sub_elapse
        else:
            dt_boxes, elapse = self.predict(img)
        return dt_boxes, elapse

