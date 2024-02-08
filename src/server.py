import argparse
import time

from starlette.requests import Request
from ray import serve
from ray.serve.handle import DeploymentHandle

from models.ocr_model.utils.inference import inference
from models.ocr_model.model.TexTeller import TexTeller


parser = argparse.ArgumentParser()
parser.add_argument('-ckpt', '--checkpoint_dir', type=str, required=True)
parser.add_argument('-tknz', '--tokenizer_dir', type=str, required=True)

parser.add_argument('-port', '--server_port', type=int, default=8000)
parser.add_argument('--num_replicas', type=int, default=1)
parser.add_argument('--ncpu_per_replica', type=float, default=1.0)
parser.add_argument('--ngpu_per_replica', type=float, default=0.0)

parser.add_argument('--use_cuda', action='store_true', default=False)
parser.add_argument('--num_beam', type=int, default=1)

# args = parser.parse_args()
args = parser.parse_args([
    '--checkpoint_dir', '/home/lhy/code/TeXify/src/models/ocr_model/model_checkpoint',
    '--tokenizer_dir', '/home/lhy/code/TeXify/src/models/tokenizer/roberta-tokenizer-550K',

    '--server_port', '9900',
    '--num_replicas', '1',
    '--ncpu_per_replica', '1.0',
    '--ngpu_per_replica', '0.0',

    # '--use_cuda',
    '--num_beam', '1'
])

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
    
    def predict(self, image_path: str) -> str:
        return inference(self.model, self.tokenizer, [image_path], self.use_cuda, self.num_beam)[0]


@serve.deployment()
class Ingress:
    def __init__(self, texteller_server: DeploymentHandle) -> None:
        self.texteller_server = texteller_server
    
    async def __call__(self, request: Request) -> str:
        msg = await request.json()
        img_path: str = msg['img_path']
        pred = await self.texteller_server.predict.remote(img_path)
        return pred


if __name__ == '__main__':
    ckpt_dir = args.checkpoint_dir
    tknz_dir = args.tokenizer_dir

    serve.start(http_options={"port": args.server_port})  # 启动一个Ray集群，端口号为9900
    texteller_server = TexTellerServer.bind(ckpt_dir, tknz_dir, use_cuda=args.use_cuda, num_beam=args.num_beam)
    ingress = Ingress.bind(texteller_server)

    ingress_handle = serve.run(ingress, route_prefix="/predict")  

    while True:
        time.sleep(1)
