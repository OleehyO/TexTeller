<div align="center">
<h1><img src="./fire.svg" width=30, height=30> 
𝚃𝚎𝚡𝚃𝚎𝚕𝚕𝚎𝚛 <img src="./fire.svg" width=30, height=30> </h1>

<p align="center">
<a href="../README.md">English</a> | 中文版本
</p>

<p align="center">
  <img src="./web_demo.gif" alt="TexTeller_demo" width=800>
</p>

</div>

TexTeller是一个基于ViT的端到端公式识别模型，可以把图片转换为对应的latex公式

TexTeller用了550K的图片-公式对进行训练(数据集可以在[这里](https://huggingface.co/datasets/OleehyO/latex-formulas)获取)，相比于[LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR)(使用了一个100K的数据集)，TexTeller具有**更强的泛化能力**以及**更高的准确率**，可以覆盖大部分的使用场景(**扫描图片，手写公式除外**)。

> 我们马上就会发布一个使用5.5M数据集进行训练的TexTeller checkpoint

## 前置条件

python=3.10

pytorch

> 注意: 只有CUDA版本>= 12.0被完全测试过，所以最好使用>= 12.0的CUDA版本

## Getting Started

1. 克隆本仓库:

    ```bash
    git clone https://github.com/OleehyO/TexTeller
    ```

2. [安装pytorch](https://pytorch.org/get-started/locally/#start-locally)后，再安装本项目的依赖包:

    ```bash
    pip install -r requirements.txt
    ```

3. 进入`TexTeller/src`目录，在终端运行以下命令进行推理:

    ```bash
    python inference.py -img "/path/to/image.{jpg,png}" 
    # use -cuda option to enable GPU inference
    #+e.g. python inference.py -img "./img.jpg" -cuda
    ```

    > 第一次运行时会在hugging face上下载所需要的checkpoints

## FAQ：无法连接到Hugging Face

默认情况下，会在Hugging Face中下载模型权重，**如果你的远端服务器无法连接到Hugging Face**，你可以通过以下命令进行加载：

1. 安装huggingface hub包

    ```bash
    pip install -U "huggingface_hub[cli]"
    ```

2. 在能连接Hugging Face的机器上下载模型权重:

    ```bash
    huggingface-cli download OleehyO/TexTeller --include "*.json" "*.bin" "*.txt" --repo-type model --local-dir "your/dir/path"
    ```

3. 把包含权重的目录上传远端服务器，然后把`TexTeller/src/models/ocr_model/model/TexTeller.py`中的`REPO_NAME = 'OleehyO/TexTeller'`修改为`REPO_NAME = 'your/dir/path'`

如果你还想在训练模型时开启evaluate，你需要提前下载metric脚本并上传远端服务器：

1. 在能连接Hugging Face的机器上下载metric脚本

    ```bash
    huggingface-cli download evaluate-metric/google_bleu --repo-type space --local-dir "your/dir/path"
    ```

2. 把这个目录上传远端服务器，并在`TexTeller/src/models/ocr_model/utils/metrics.py`中把`evaluate.load('google_bleu')`改为`evaluate.load('your/dir/path/google_bleu.py')`

## Web Demo

要想启动web demo，你需要先进入 `TexTeller/src` 目录，然后运行以下命令

```bash
./start_web.sh
```

然后在浏览器里输入`http://localhost:8501`就可以看到web demo

> 你可以改变`start_web.sh`的默认配置， 例如使用GPU进行推理(e.g. `USE_CUDA=True`) 或者增加beams的数量(e.g. `NUM_BEAM=3`)来获得更高的精确度

## API

我们使用[ray serve](https://github.com/ray-project/ray)来对外提供一个TexTeller的API接口，通过使用这个接口，你可以把TexTeller整合到自己的项目里。要想启动server，你需要先进入`TexTeller/src`目录然后运行以下命令:

```bash
python server.py  # default settings
```

你可以给`server.py`传递以下参数来改变server的推理设置(e.g. `python server.py --use_gpu` 来启动GPU推理):

| Argument | Description |
| --- | --- |
| `-ckpt` | Path to the checkpoint file to load, default is TexTeller pretrained model. |
| `-tknz` | Path to the tokenizer, default is TexTeller tokenizer. |
| `-port` | Port number to run the server on, *default is 8000*. |
| `--use_gpu` | Whether to use GPU for inference. |
| `--num_beams` | Number of beams to use for beam search decoding, *default is 1*. |
| `--num_replicas` | Number of replicas to run the server on, *default is 1*. You can use this to get higher throughput. |
| `--ncpu_per_replica` | Number of CPU cores to use per replica, *default is 1*. |
| `--ngpu_per_replica` | Number of GPUs to use per replica, *default is 1*. You can set this to 0~1 to run multiple replicas on a single GPU(if --num_replicas 2, --ngpu_per_replica 0.7, then 2 gpus are required) |

> 一个客户端demo可以在`TexTeller/client/demo.py`找到，你可以参考`demo.py`来给server发送请求

## Training

### Dataset

我们在`TexTeller/src/models/ocr_model/train/dataset`目录中提供了一个数据集的例子，你可以把自己的图片放在`images`目录然后在`formulas.jsonl`中为每张图片标注对应的公式。

准备好数据集后，你需要在`.../dataset/loader.py`中把 **`DIR_URL`变量改成你自己数据集的路径**

### Retrain the tokenizer

如果你使用了不一样的数据集，你可能需要重新训练tokenizer来得到一个不一样的字典。配置好数据集后，可以通过以下命令来训练自己的tokenizer：

1. 在`TexTeller/src/models/tokenizer/train.py`中，修改`new_tokenizer.save_pretrained('./your_dir_name')`为你自定义的输出目录
    > 如果要用一个不一样大小的字典(默认1W个token)，你需要在 `TexTeller/src/models/globals.py`中修改`VOCAB_SIZE`变量

2. **在 `TexTeller/src` 目录下**运行以下命令:

    ```bash
    python -m models.tokenizer.train
    ```

### Train the model

要想训练模型, 你需要在`TexTeller/src`目录下运行以下命令：

```bash
python -m models.ocr_model.train.train
```

你可以在`TexTeller/src/models/ocr_model/train/train.py`中设置自己的tokenizer和checkpoint路径（请参考`train.py`）。如果你使用了与TexTeller一样的架构和相同的字典，你还可以用自己的数据集来微调TexTeller的默认权重。

在`TexTeller/src/globals.py`和`TexTeller/src/models/ocr_model/train/train_args.py`中，你可以改变模型的架构以及训练的超参数。

> 我们的训练脚本使用了[Hugging Face Transformers](https://github.com/huggingface/transformers)库, 所以你可以参考他们提供的[文档](https://huggingface.co/docs/transformers/v4.32.1/main_classes/trainer#transformers.TrainingArguments)来获取更多训练参数的细节以及配置。

## To-Do

- [ ] 使用更大的数据集来训练模型(5.5M样本，即将发布)

- [ ] 推理加速

- [ ] ...

## Acknowledgements

Thanks to [LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR) which has brought me a lot of inspiration, and [im2latex-100K](https://zenodo.org/records/56198#.V2px0jXT6eA) which enriches our dataset.