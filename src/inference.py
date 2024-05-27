import os
import argparse
import cv2 as cv

from pathlib import Path
from onnxruntime import InferenceSession
from models.thrid_party.paddleocr.infer import predict_det, predict_rec
from models.thrid_party.paddleocr.infer import utility

from models.utils import mix_inference
from models.ocr_model.utils.to_katex import to_katex
from models.ocr_model.utils.inference import inference as latex_inference

from models.ocr_model.model.TexTeller import TexTeller
from models.det_model.inference import PredictConfig


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-img', 
        type=str, 
        required=True,
        help='path to the input image'
    )
    parser.add_argument(
        '--inference-mode', 
        type=str,
        default='cpu',
        help='Inference mode, select one of cpu, cuda, or mps'
    )
    parser.add_argument(
        '--num-beam', 
        type=int,
        default=1,
        help='number of beam search for decoding'
    )
    parser.add_argument(
        '-mix', 
        action='store_true',
        help='use mix mode'
    )
    
    args = parser.parse_args()
    
    # You can use your own checkpoint and tokenizer path.
    print('Loading model and tokenizer...')
    latex_rec_model = TexTeller.from_pretrained()
    tokenizer = TexTeller.get_tokenizer()
    print('Model and tokenizer loaded.')

    img_path = args.img
    img = cv.imread(img_path)
    print('Inference...')
    if not args.mix:
        res = latex_inference(latex_rec_model, tokenizer, [img], args.inference_mode, args.num_beam)
        res = to_katex(res[0])
        print(res)
    else:
        infer_config = PredictConfig("./models/det_model/model/infer_cfg.yml")
        latex_det_model = InferenceSession("./models/det_model/model/rtdetr_r50vd_6x_coco.onnx")

        use_gpu = args.inference_mode == 'cuda'
        SIZE_LIMIT = 20 * 1024 * 1024
        det_model_dir =  "./models/thrid_party/paddleocr/checkpoints/det/default_model.onnx"
        rec_model_dir =  "./models/thrid_party/paddleocr/checkpoints/rec/default_model.onnx"
        # The CPU inference of the detection model will be faster than the GPU inference (in onnxruntime)
        det_use_gpu = False
        rec_use_gpu = use_gpu and not (os.path.getsize(rec_model_dir) < SIZE_LIMIT)

        paddleocr_args = utility.parse_args()
        paddleocr_args.use_onnx = True
        paddleocr_args.det_model_dir = det_model_dir
        paddleocr_args.rec_model_dir = rec_model_dir

        paddleocr_args.use_gpu = det_use_gpu
        detector = predict_det.TextDetector(paddleocr_args)
        paddleocr_args.use_gpu = rec_use_gpu
        recognizer = predict_rec.TextRecognizer(paddleocr_args)
        
        lang_ocr_models = [detector, recognizer]
        latex_rec_models = [latex_rec_model, tokenizer]
        res = mix_inference(img_path, infer_config, latex_det_model, lang_ocr_models, latex_rec_models, args.inference_mode, args.num_beam)
        print(res)
