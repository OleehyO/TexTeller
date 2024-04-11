import os
import yaml
import argparse
import numpy as np
import glob
from onnxruntime import InferenceSession
from tqdm import tqdm

from models.det_model.preprocess import Compose
import cv2

# 注意：文件名要标准，最好都用下划线

# Global dictionary
SUPPORT_MODELS = {
    'YOLO', 'PPYOLOE', 'RCNN', 'SSD', 'Face', 'FCOS', 'SOLOv2', 'TTFNet',
    'S2ANet', 'JDE', 'FairMOT', 'DeepSORT', 'GFL', 'PicoDet', 'CenterNet',
    'TOOD', 'RetinaNet', 'StrongBaseline', 'STGCN', 'YOLOX', 'HRNet', 
    'DETR'
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--infer_cfg", type=str, help="infer_cfg.yml",
                    default="./models/det_model/model/infer_cfg.yml"
                    )
parser.add_argument('--onnx_file', type=str, help="onnx model file path",
                    default="./models/det_model/model/rtdetr_r50vd_6x_coco.onnx"
                    )
parser.add_argument("--image_dir", type=str)
parser.add_argument("--image_file", type=str, default='/data/ljm/TexTeller/src/Tr00_0001015-page02.jpg')
parser.add_argument("--imgsave_dir", type=str, 
                    default="."
                    )

def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--image_file or --image_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)

    assert len(images) > 0, "no image found in {}".format(infer_dir)
    print("Found {} inference images in total.".format(len(images)))

    return images


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

        # 预定义颜色池
        color_pool = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        # 根据label_list动态生成颜色映射
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
            # 获取类别名
            label = infer_config.label_list[int(cls_id)]
            # 根据类别名获取颜色
            color = infer_config.colors[label]
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
            cv2.putText(image, "{}: {:.2f}".format(label, score),
                        (int(xmin), int(ymin - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def predict_image(infer_config, predictor, img_list):
    # load preprocess transforms
    transforms = Compose(infer_config.preprocess_infos)
    errImgList = []

    # Check and create subimg_save_dir if not exist
    subimg_save_dir = os.path.join(FLAGS.imgsave_dir, 'subimages')
    os.makedirs(subimg_save_dir, exist_ok=True)

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

        outputs = predictor.run(output_names=None, input_feed=inputs)

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
                subimg = img[int(ymin):int(ymax), int(xmin):int(xmax)]
                subimg_filename = f"{os.path.splitext(os.path.basename(img_path))[0]}_{label}_{xmin:.2f}_{ymin:.2f}_{xmax:.2f}_{ymax:.2f}.jpg"
                subimg_path = os.path.join(subimg_save_dir, subimg_filename)
                cv2.imwrite(subimg_path, subimg)
                subimg_counter += 1

        # Draw bounding boxes and save the image with bounding boxes
        img_with_bbox = draw_bbox(img, np.array(outputs[0]), infer_config)
        output_dir = FLAGS.imgsave_dir
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "output_" + os.path.basename(img_path))
        cv2.imwrite(output_file, img_with_bbox)

    print("ErrorImgs:")
    print(errImgList)

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    # load image list
    img_list = get_test_images(FLAGS.image_dir, FLAGS.image_file)
    # load predictor
    predictor = InferenceSession(FLAGS.onnx_file)
    # load infer config
    infer_config = PredictConfig(FLAGS.infer_cfg)

    predict_image(infer_config, predictor, img_list)
