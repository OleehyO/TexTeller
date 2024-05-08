import os
import time
import yaml
import numpy as np
import cv2

from tqdm import tqdm
from typing import List
from .preprocess import Compose
from .Bbox import Bbox


# Global dictionary
SUPPORT_MODELS = {
    'YOLO', 'PPYOLOE', 'RCNN', 'SSD', 'Face', 'FCOS', 'SOLOv2', 'TTFNet',
    'S2ANet', 'JDE', 'FairMOT', 'DeepSORT', 'GFL', 'PicoDet', 'CenterNet',
    'TOOD', 'RetinaNet', 'StrongBaseline', 'STGCN', 'YOLOX', 'HRNet', 
    'DETR'
}


class PredictConfig(object):
    """set config of preprocess, postprocess and visualize
    Args:
        infer_config (str): path of infer_cfg.yml
    """

    def __init__(self, infer_config):
        # parsing Yaml config for Preprocess
        with open(infer_config) as f:
            yml_conf = yaml.safe_load(f)
        self.check_model(yml_conf)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.label_list = yml_conf['label_list']
        self.use_dynamic_shape = yml_conf['use_dynamic_shape']
        self.draw_threshold = yml_conf.get("draw_threshold", 0.5)
        self.mask = yml_conf.get("mask", False)
        self.tracker = yml_conf.get("tracker", None)
        self.nms = yml_conf.get("NMS", None)
        self.fpn_stride = yml_conf.get("fpn_stride", None)

        color_pool = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        self.colors = {label: color_pool[i % len(color_pool)] for i, label in enumerate(self.label_list)}

        if self.arch == 'RCNN' and yml_conf.get('export_onnx', False):
            print(
                'The RCNN export model is used for ONNX and it only supports batch_size = 1'
            )
        self.print_config()

    def check_model(self, yml_conf):
        """
        Raises:
            ValueError: loaded model not in supported model type
        """
        for support_model in SUPPORT_MODELS:
            if support_model in yml_conf['arch']:
                return True
        raise ValueError("Unsupported arch: {}, expect {}".format(yml_conf[
            'arch'], SUPPORT_MODELS))

    def print_config(self):
        print('-----------  Model Configuration -----------')
        print('%s: %s' % ('Model Arch', self.arch))
        print('%s: ' % ('Transform Order'))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))
        print('--------------------------------------------')


def draw_bbox(image, outputs, infer_config):
    for output in outputs:
        cls_id, score, xmin, ymin, xmax, ymax = output
        if score > infer_config.draw_threshold:
            label = infer_config.label_list[int(cls_id)]
            color = infer_config.colors[label]
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
            cv2.putText(image, "{}: {:.2f}".format(label, score),
                        (int(xmin), int(ymin - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def predict_image(imgsave_dir, infer_config, predictor, img_list):
    # load preprocess transforms
    transforms = Compose(infer_config.preprocess_infos)
    errImgList = []

    # Check and create subimg_save_dir if not exist
    subimg_save_dir = os.path.join(imgsave_dir, 'subimages')
    os.makedirs(subimg_save_dir, exist_ok=True)

    first_image_skipped = False
    total_time = 0
    num_images = 0
    # predict image
    for img_path in tqdm(img_list):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}. Skipping...")
            errImgList.append(img_path)
            continue

        inputs = transforms(img_path)
        inputs_name = [var.name for var in predictor.get_inputs()]
        inputs = {k: inputs[k][None, ] for k in inputs_name}

        # Start timing
        start_time = time.time()

        outputs = predictor.run(output_names=None, input_feed=inputs)

        # Stop timing
        end_time = time.time()
        inference_time = end_time - start_time
        if not first_image_skipped:
            first_image_skipped = True
        else:
            total_time += inference_time
            num_images += 1
        print(f"ONNXRuntime predict time for {os.path.basename(img_path)}: {inference_time:.4f} seconds")

        print("ONNXRuntime predict: ")
        if infer_config.arch in ["HRNet"]:
            print(np.array(outputs[0]))
        else:
            bboxes = np.array(outputs[0])
            for bbox in bboxes:
                if bbox[0] > -1 and bbox[1] > infer_config.draw_threshold:
                    print(f"{int(bbox[0])} {bbox[1]} "
                          f"{bbox[2]} {bbox[3]} {bbox[4]} {bbox[5]}")

        # Save the subimages (crop from the original image)
        subimg_counter = 1
        for output in np.array(outputs[0]):
            cls_id, score, xmin, ymin, xmax, ymax = output
            if score > infer_config.draw_threshold:
                label = infer_config.label_list[int(cls_id)]
                subimg = img[int(max(ymin, 0)):int(ymax), int(max(xmin, 0)):int(xmax)]
                if len(subimg) == 0:
                    continue

                subimg_filename = f"{os.path.splitext(os.path.basename(img_path))[0]}_{label}_{xmin:.2f}_{ymin:.2f}_{xmax:.2f}_{ymax:.2f}.jpg"
                subimg_path = os.path.join(subimg_save_dir, subimg_filename)
                cv2.imwrite(subimg_path, subimg)
                subimg_counter += 1

        # Draw bounding boxes and save the image with bounding boxes
        img_with_mask = img.copy()
        for output in np.array(outputs[0]):
            cls_id, score, xmin, ymin, xmax, ymax = output
            if score > infer_config.draw_threshold:
                cv2.rectangle(img_with_mask, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 255), -1) # 盖白
        
        img_with_bbox = draw_bbox(img, np.array(outputs[0]), infer_config)

        output_dir = imgsave_dir
        os.makedirs(output_dir, exist_ok=True)
        draw_box_dir = os.path.join(output_dir, 'draw_box')
        mask_white_dir = os.path.join(output_dir, 'mask_white')
        os.makedirs(draw_box_dir, exist_ok=True)
        os.makedirs(mask_white_dir, exist_ok=True)

        output_file_mask = os.path.join(mask_white_dir, os.path.basename(img_path))
        output_file_bbox = os.path.join(draw_box_dir, os.path.basename(img_path))
        cv2.imwrite(output_file_mask, img_with_mask)
        cv2.imwrite(output_file_bbox, img_with_bbox)

    avg_time_per_image = total_time / num_images if num_images > 0 else 0
    print(f"Total inference time for {num_images} images: {total_time:.4f} seconds")
    print(f"Average time per image: {avg_time_per_image:.4f} seconds")
    print("ErrorImgs:")
    print(errImgList)


def predict(img_path: str, predictor, infer_config) -> List[Bbox]:
    transforms = Compose(infer_config.preprocess_infos)
    inputs = transforms(img_path)
    inputs_name = [var.name for var in predictor.get_inputs()]
    inputs = {k: inputs[k][None, ] for k in inputs_name}

    outputs = predictor.run(output_names=None, input_feed=inputs)[0]
    res = []
    for output in outputs:
        cls_name = infer_config.label_list[int(output[0])]
        score = output[1]
        xmin = int(max(output[2], 0))
        ymin = int(max(output[3], 0))
        xmax = int(output[4])
        ymax = int(output[5])
        if score > infer_config.draw_threshold:
            res.append(Bbox(xmin, ymin, ymax - ymin, xmax - xmin, cls_name, score))

    return res
