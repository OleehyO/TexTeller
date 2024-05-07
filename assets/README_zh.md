📄 <a href="../README.md">English</a> | 中文

<div align="center">
    <h1>
        <img src="./fire.svg" width=30, height=30> 
        𝚃𝚎𝚡𝚃𝚎𝚕𝚕𝚎𝚛
        <img src="./fire.svg" width=30, height=30> 
    </h1>
    <p align="center">
        🤗 <a href="https://huggingface.co/OleehyO/TexTeller">Hugging Face</a>
    </p>
    <!-- <p align="center">
        <img src="./web_demo.gif" alt="TexTeller_demo" width=800>
    </p> -->
</div>

https://github.com/OleehyO/TexTeller/assets/56267907/fb17af43-f2a5-47ce-ad1d-101db5fd7fbb

TexTeller是一个基于ViT的端到端公式识别模型，可以把图片转换为对应的latex公式

TexTeller用了7.5M的图片-公式对进行训练(数据集可以在[这里](https://huggingface.co/datasets/OleehyO/latex-formulas)获取)，相比于[LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR)(使用了一个100K的数据集)，TexTeller具有**更强的泛化能力**以及**更高的准确率**，可以覆盖大部分的使用场景(**扫描图片，手写公式除外**)。

## 🔄 变更信息

* 📮[2024-03-25] TexTeller2.0发布！TexTeller2.0的训练数据增大到了7.5M(相较于TexTeller1.0**增加了~15倍**并且数据质量也有所改善)。训练后的TexTeller2.0在测试集中展现出了**更加优越的性能**，尤其在生僻符号、复杂多行、矩阵的识别场景中。

  > 在[这里](./test.pdf)有更多的测试图片以及各家识别模型的横向对比。
  >
* 📮[2024-04-12] 训练了**公式检测模型**，从而增加了对整个文档进行公式检测+公式识别（整图推理）的功能！

* 📮[2024-05-02] 支持**中英文-公式混合识别**。

## 🔑 前置条件

python=3.10

[pytorch](https://pytorch.org/get-started/locally/)

> 只有CUDA版本>= 12.0被完全测试过，所以最好使用>= 12.0的CUDA版本

## 🚀 开搞

1. 克隆本仓库:

   ```bash
   git clone https://github.com/OleehyO/TexTeller
   ```

2. [安装pytorch](https://pytorch.org/get-started/locally/#start-locally)

3. 安装本项目的依赖包:

   ```bash
   pip install -r requirements.txt
   ```

4. 进入 `TexTeller/src`目录，在终端运行以下命令进行推理:

   ```bash
    python inference.py -img "/path/to/image.{jpg,png}" 
    # use --inference-mode option to enable GPU(cuda or mps) inference
    #+e.g. python inference.py -img "./img.jpg" --inference-mode cuda
    # use -mix option to enable mixed text and formula recognition
    #+e.g. python inferene.py -img "./img.jpg" -mix -lang "en"
   ```

   > 第一次运行时会在Hugging Face上下载所需要的权重

> [!IMPORTANT]
> 如果使用文字-公式混合识别，需要[下载公式检测模型的权重](https://github.com/OleehyO/TexTeller/blob/main/assets/README_zh.md#%E4%B8%8B%E8%BD%BD%E6%9D%83%E9%87%8D)

## ❓ 常见问题：无法连接到Hugging Face

默认情况下，会在Hugging Face中下载模型权重，**如果你的远端服务器无法连接到Hugging Face**，你可以通过以下命令进行加载：

1. 安装huggingface hub包

   ```bash
   pip install -U "huggingface_hub[cli]"
   ```

2. 在能连接Hugging Face的机器上下载模型权重:

   ```bash
   huggingface-cli download \
       OleehyO/TexTeller \
       --repo-type model \
       --local-dir "your/dir/path" \
       --local-dir-use-symlinks False
   ```

3. 把包含权重的目录上传远端服务器，然后把 `TexTeller/src/models/ocr_model/model/TexTeller.py`中的 `REPO_NAME = 'OleehyO/TexTeller'`修改为 `REPO_NAME = 'your/dir/path'`

如果你还想在训练模型时开启evaluate，你需要提前下载metric脚本并上传远端服务器：

1. 在能连接Hugging Face的机器上下载metric脚本

   ```bash
   huggingface-cli download \
       evaluate-metric/google_bleu \
       --repo-type space \
       --local-dir "your/dir/path" \
       --local-dir-use-symlinks False
   ```

2. 把这个目录上传远端服务器，并在 `TexTeller/src/models/ocr_model/utils/metrics.py`中把 `evaluate.load('google_bleu')`改为 `evaluate.load('your/dir/path/google_bleu.py')`

## 🌐 网页演示

进入 `TexTeller/src` 目录，运行以下命令

```bash
./start_web.sh
```

在浏览器里输入 `http://localhost:8501`就可以看到web demo

> [!NOTE]
> 对于Windows用户, 请运行 `start_web.bat`文件.

## 🧠 整图推理

TexTeller还支持对整张图片进行**公式检测+公式识别**，从而对整图公式进行检测，然后进行批公式识别。

### 下载权重

根据[这里的链接](https://huggingface.co/TonyLee1256/texteller_det/resolve/main/rtdetr_r50vd_6x_coco.onnx?download=true)把模型权重下载到`src/models/det_model/model`即可

> TexTeller的公式检测模型在3415张中文教材数据(130+版式)和8272张[IBEM数据集](https://zenodo.org/records/4757865)上，共11867张图片上训练得到.

### 公式检测

 `TexTeller/src`目录下运行以下命令

```bash
python infer_det.py
```

对整张图中的所有公式进行检测，结果保存在 `TexTeller/src/subimages`

<div align="center">
    <img src="det_rec.png" width=400> 
</div>

### 公式批识别

在进行**公式检测后**， `TexTeller/src`目录下运行以下命令

```shell
python rec_infer_from_crop_imgs.py
```

会基于上一步公式检测的结果，对裁剪出的所有公式进行批量识别，将识别结果在 `TexTeller/src/results`中保存为txt文件。

## 📡 API调用

我们使用[ray serve](https://github.com/ray-project/ray)来对外提供一个TexTeller的API接口，通过使用这个接口，你可以把TexTeller整合到自己的项目里。要想启动server，你需要先进入 `TexTeller/src`目录然后运行以下命令:

```bash
python server.py 
```

| 参数 | 描述 |
| --- | --- |
| `-ckpt` | 权重文件的路径，*默认为TexTeller的预训练权重*。|
| `-tknz` | 分词器的路径，*默认为TexTeller的分词器*。|
| `-port` | 服务器的服务端口，*默认是8000*。|
| `--inference-mode` | 是否使用GPU(cuda或mps)推理，*默认为CPU*。|
| `--num_beams` | beam search的beam数量，*默认是1*。|
| `--num_replicas` | 在服务器上运行的服务副本数量，*默认1个副本*。你可以使用更多的副本来获取更大的吞吐量。|
| `--ncpu_per_replica` | 每个服务副本所用的CPU核心数，*默认为1*。|
| `--ngpu_per_replica` | 每个服务副本所用的GPU数量，*默认为1*。你可以把这个值设置成 0~1之间的数，这样会在一个GPU上运行多个服务副本来共享GPU，从而提高GPU的利用率。(注意，如果 --num_replicas 2, --ngpu_per_replica 0.7, 那么就必须要有2个GPU可用) |

> [!NOTE]
> 一个客户端demo可以在 `TexTeller/client/demo.py`找到，你可以参考 `demo.py`来给server发送请求

## 🏋️‍♂️ 训练

### 数据集

我们在 `TexTeller/src/models/ocr_model/train/dataset`目录中提供了一个数据集的例子，你可以把自己的图片放在 `images`目录然后在 `formulas.jsonl`中为每张图片标注对应的公式。

准备好数据集后，你需要在 `.../dataset/loader.py`中把 **`DIR_URL`变量改成你自己数据集的路径**

### 重新训练分词器

如果你使用了不一样的数据集，你可能需要重新训练tokenizer来得到一个不一样的字典。配置好数据集后，可以通过以下命令来训练自己的tokenizer：

1. 在 `TexTeller/src/models/tokenizer/train.py`中，修改 `new_tokenizer.save_pretrained('./your_dir_name')`为你自定义的输出目录

   > 注意：如果要用一个不一样大小的字典(默认1W个token)，你需要在 `TexTeller/src/models/globals.py`中修改 `VOCAB_SIZE`变量
   >
2. **在 `TexTeller/src` 目录下**运行以下命令:

   ```bash
   python -m models.tokenizer.train
   ```

### 训练模型

1. 修改`src/train_config.yaml`中的`num_processes`为训练用的显卡数(默认为1)

2. 在`TexTeller/src`目录下运行以下命令：

   ```bash
   accelerate launch --config_file ./train_config.yaml -m models.ocr_model.train.train
   ```

你可以在 `TexTeller/src/models/ocr_model/train/train.py`中设置自己的tokenizer和checkpoint路径（请参考 `train.py`）。如果你使用了与TexTeller一样的架构和相同的字典，你还可以用自己的数据集来微调TexTeller的默认权重。

在 `TexTeller/src/globals.py`和 `TexTeller/src/models/ocr_model/train/train_args.py`中，你可以改变模型的架构以及训练的超参数。

> [!NOTE]
> 我们的训练脚本使用了[Hugging Face Transformers](https://github.com/huggingface/transformers)库, 所以你可以参考他们提供的[文档](https://huggingface.co/docs/transformers/v4.32.1/main_classes/trainer#transformers.TrainingArguments)来获取更多训练参数的细节以及配置。

## 🚧 不足

* 不支持扫描图片以及PDF文档识别
* 不支持手写体公式

## 📅 计划

- [X] ~~使用更大的数据集来训练模型~~
- [ ] 扫描图片识别
- [ ] PDF文档识别 + 中英文场景支持
- [ ] 推理加速
- [ ] ...

## ⭐️ 观星曲线

[![Stargazers over time](https://starchart.cc/OleehyO/TexTeller.svg?variant=adaptive)](https://starchart.cc/OleehyO/TexTeller)

## 💖 感谢

Thanks to [LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR) which has brought me a lot of inspiration, and [im2latex-100K](https://zenodo.org/records/56198#.V2px0jXT6eA) which enriches our dataset.

## 👥 贡献者

<a href="https://github.com/OleehyO/TexTeller/graphs/contributors">
   <a href="https://github.com/OleehyO/TexTeller/graphs/contributors">
      <img src="https://contrib.rocks/image?repo=OleehyO/TexTeller" />
   </a>
</a>
