import os
import sys
import argparse
import cv2 as cv

from pathlib import Path
from onnxruntime import InferenceSession
from paddleocr import PaddleOCR

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
    parser.add_argument(
        '-lang', 
        type=str,
        default='None'
    )
    
    args = parser.parse_args()
    if args.mix and args.lang == "None":
        print("When -mix is set, -lang must be set (support: ['zh', 'en'])")
        sys.exit(-1)
    elif args.mix and args.lang not in ['zh', 'en']:
        print(f"language support: ['zh', 'en'] (invalid: {args.lang})")
        sys.exit(-1)
    
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
        text_ocr_model = PaddleOCR(
            use_angle_cls=False, lang='ch', use_gpu=use_gpu,
            det_model_dir="./models/text_ocr_model/infer_models/ch_PP-OCRv4_det_server_infer",
            rec_model_dir="./models/text_ocr_model/infer_models/ch_PP-OCRv4_rec_server_infer",
            det_limit_type='max',
            det_limit_side_len=1280,
            use_dilation=True,
            det_db_score_mode="slow",
        ) # need to run only once to load model into memory

        detector = text_ocr_model.text_detector
        recognizer = text_ocr_model.text_recognizer
        
        lang_ocr_models = [detector, recognizer]
        latex_rec_models = [latex_rec_model, tokenizer]
        res = mix_inference(img_path, args.lang , infer_config, latex_det_model, lang_ocr_models, latex_rec_models, args.inference_mode, args.num_beam)
        print(res)
