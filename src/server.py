import argparse
import tempfile
import time
import numpy as np
import cv2

from pathlib import Path
from starlette.requests import Request
from ray import serve
from ray.serve.handle import DeploymentHandle
from onnxruntime import InferenceSession

from models.ocr_model.utils.inference import inference as rec_inference
from models.det_model.inference import predict as det_inference
from models.ocr_model.model.TexTeller import TexTeller
from models.det_model.inference import PredictConfig
from models.ocr_model.utils.to_katex import to_katex


parser = argparse.ArgumentParser()
parser.add_argument(
    '-ckpt', '--checkpoint_dir', type=str
)
parser.add_argument(
    '-tknz', '--tokenizer_dir', type=str
)
parser.add_argument('-port', '--server_port', type=int, default=8000)
parser.add_argument('--num_replicas', type=int, default=1)
parser.add_argument('--ncpu_per_replica', type=float, default=1.0)
parser.add_argument('--ngpu_per_replica', type=float, default=0.0)

parser.add_argument('--inference-mode', type=str, default='cpu')
parser.add_argument('--num_beams', type=int, default=1)

args = parser.parse_args()
if args.ngpu_per_replica > 0 and not args.inference_mode == 'cuda':
    raise ValueError("--inference-mode must be cuda or mps if ngpu_per_replica > 0")
    

@serve.deployment(
    num_replicas=args.num_replicas, 
    ray_actor_options={
        "num_cpus": args.ncpu_per_replica, 
        "num_gpus": args.ngpu_per_replica
    }
)
class TexTellerRecServer:
    def __init__(
        self, 
        checkpoint_path: str, 
        tokenizer_path: str, 
        inf_mode: str = 'cpu',
        num_beams: int = 1
    ) -> None:
        self.model = TexTeller.from_pretrained(checkpoint_path)
        self.tokenizer = TexTeller.get_tokenizer(tokenizer_path)
        self.inf_mode = inf_mode
        self.num_beams = num_beams

        self.model = self.model.to(inf_mode) if inf_mode != 'cpu' else self.model
    
    def predict(self, image_nparray) -> str:
        return to_katex(rec_inference(
            self.model, self.tokenizer, [image_nparray],
            accelerator=self.inf_mode, num_beams=self.num_beams
        )[0])


@serve.deployment(num_replicas=args.num_replicas)
class TexTellerDetServer:
    def __init__(
        self
    ):
        self.infer_config = PredictConfig("./models/det_model/model/infer_cfg.yml")
        self.latex_det_model = InferenceSession("./models/det_model/model/rtdetr_r50vd_6x_coco.onnx")

    async def predict(self, image_nparray) -> str:
        with tempfile.TemporaryDirectory() as temp_dir:
            img_path = f"{temp_dir}/temp_image.jpg"
            cv2.imwrite(img_path, image_nparray)
            
            latex_bboxes = det_inference(img_path, self.latex_det_model, self.infer_config)
            return latex_bboxes


@serve.deployment()
class Ingress:
    def __init__(self, det_server: DeploymentHandle, rec_server: DeploymentHandle) -> None:
        self.det_server = det_server
        self.texteller_server = rec_server
    
    async def __call__(self, request: Request) -> str:
        request_path = request.url.path
        form   = await request.form()
        img_rb = await form['img'].read()

        img_nparray = np.frombuffer(img_rb, np.uint8)
        img_nparray = cv2.imdecode(img_nparray, cv2.IMREAD_COLOR)
        img_nparray = cv2.cvtColor(img_nparray, cv2.COLOR_BGR2RGB)

        if request_path.startswith("/fdet"):
            if self.det_server == None:
                return "[ERROR] rtdetr_r50vd_6x_coco.onnx not found."
            pred = await self.det_server.predict.remote(img_nparray)
            return pred

        elif request_path.startswith("/frec"):
            pred = await self.texteller_server.predict.remote(img_nparray)
            return pred

        else:
            return "[ERROR] Invalid request path"


if __name__ == '__main__':
    ckpt_dir = args.checkpoint_dir
    tknz_dir = args.tokenizer_dir

    serve.start(http_options={"host": "0.0.0.0", "port": args.server_port})
    rec_server = TexTellerRecServer.bind(
        ckpt_dir, tknz_dir, 
        inf_mode=args.inference_mode,
        num_beams=args.num_beams
    )
    det_server = None
    if Path('./models/det_model/model/rtdetr_r50vd_6x_coco.onnx').exists():
        det_server = TexTellerDetServer.bind()
    ingress = Ingress.bind(det_server, rec_server)

    # ingress_handle = serve.run(ingress, route_prefix="/predict")  
    ingress_handle = serve.run(ingress, route_prefix="/") 

    while True:
        time.sleep(1)
