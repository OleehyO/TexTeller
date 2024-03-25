import argparse
import time
import numpy as np
import cv2

from starlette.requests import Request
from ray import serve
from ray.serve.handle import DeploymentHandle

from models.ocr_model.utils.inference import inference
from models.ocr_model.model.TexTeller import TexTeller


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

parser.add_argument('--use_cuda', action='store_true', default=False)
parser.add_argument('--num_beam', type=int, default=1)

args = parser.parse_args()
if args.ngpu_per_replica > 0 and not args.use_cuda:
    raise ValueError("use_cuda must be True if ngpu_per_replica > 0")
    

@serve.deployment(
    num_replicas=args.num_replicas, 
    ray_actor_options={
        "num_cpus": args.ncpu_per_replica, 
        "num_gpus": args.ngpu_per_replica
    }
)
class TexTellerServer:
    def __init__(
        self, 
        checkpoint_path: str, 
        tokenizer_path: str, 
        use_cuda: bool = False,
        num_beam: int = 1
    ) -> None:
        self.model = TexTeller.from_pretrained(checkpoint_path)
        self.tokenizer = TexTeller.get_tokenizer(tokenizer_path)
        self.use_cuda = use_cuda
        self.num_beam = num_beam

        self.model = self.model.to('cuda') if use_cuda else self.model
    
    def predict(self, image_nparray) -> str:
        return inference(self.model, self.tokenizer, [image_nparray], self.use_cuda, self.num_beam)[0]


@serve.deployment()
class Ingress:
    def __init__(self, texteller_server: DeploymentHandle) -> None:
        self.texteller_server = texteller_server
    
    async def __call__(self, request: Request) -> str:
        form   = await request.form()
        img_rb = await form['img'].read()

        img_nparray = np.frombuffer(img_rb, np.uint8)
        img_nparray = cv2.imdecode(img_nparray, cv2.IMREAD_COLOR)
        img_nparray = cv2.cvtColor(img_nparray, cv2.COLOR_BGR2RGB)
        pred = await self.texteller_server.predict.remote(img_nparray)
        return pred


if __name__ == '__main__':
    ckpt_dir = args.checkpoint_dir
    tknz_dir = args.tokenizer_dir

    serve.start(http_options={"port": args.server_port})
    texteller_server = TexTellerServer.bind(ckpt_dir, tknz_dir, use_cuda=args.use_cuda, num_beam=args.num_beam)
    ingress = Ingress.bind(texteller_server)

    ingress_handle = serve.run(ingress, route_prefix="/predict")  

    while True:
        time.sleep(1)
