import os
import argparse
import glob
import subprocess

import onnxruntime
from pathlib import Path

from models.det_model.inference import PredictConfig, predict_image


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--infer_cfg", type=str, help="infer_cfg.yml",
                    default="./models/det_model/model/infer_cfg.yml")
parser.add_argument('--onnx_file', type=str, help="onnx model file path",
                    default="./models/det_model/model/rtdetr_r50vd_6x_coco.onnx")
parser.add_argument("--image_dir", type=str, default='./testImgs')
parser.add_argument("--image_file", type=str)
parser.add_argument("--imgsave_dir", type=str, default="./detect_results")
parser.add_argument('--use_gpu', action='store_true', help='Whether to use GPU for inference', default=True)


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

def download_file(url, filename):
    print(f"Downloading {filename}...")
    subprocess.run(["wget", "-q", "--show-progress", "-O", filename, url], check=True)
    print("Download complete.")

if __name__ == '__main__':
    cur_path = os.getcwd()
    script_dirpath = Path(__file__).resolve().parent
    os.chdir(script_dirpath)

    FLAGS = parser.parse_args()

    if not os.path.exists(FLAGS.infer_cfg):
        infer_cfg_url = "https://huggingface.co/TonyLee1256/texteller_det/resolve/main/infer_cfg.yml?download=true"
        download_file(infer_cfg_url, FLAGS.infer_cfg)

    if not os.path.exists(FLAGS.onnx_file):
        onnx_file_url = "https://huggingface.co/TonyLee1256/texteller_det/resolve/main/rtdetr_r50vd_6x_coco.onnx?download=true"
        download_file(onnx_file_url, FLAGS.onnx_file)
    
    # load image list
    img_list = get_test_images(FLAGS.image_dir, FLAGS.image_file)

    if FLAGS.use_gpu:
        predictor = onnxruntime.InferenceSession(FLAGS.onnx_file, providers=['CUDAExecutionProvider'])
    else:
        predictor = onnxruntime.InferenceSession(FLAGS.onnx_file, providers=['CPUExecutionProvider'])
    # load infer config
    infer_config = PredictConfig(FLAGS.infer_cfg)

    predict_image(FLAGS.imgsave_dir, infer_config, predictor, img_list)

    os.chdir(cur_path)
