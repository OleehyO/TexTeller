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

TexTeller用了~~550K~~7.5M的图片-公式对进行训练(数据集可以在[这里](https://huggingface.co/datasets/OleehyO/latex-formulas)获取)，相比于[LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR)(使用了一个100K的数据集)，TexTeller具有**更强的泛化能力**以及**更高的准确率**，可以覆盖大部分的使用场景(**扫描图片，手写公式除外**)。

> ~~我们马上就会发布一个使用7.5M数据集进行训练的TexTeller checkpoint~~

## 🔄 变更信息

* 📮[2024-03-25] TexTeller2.0发布！TexTeller2.0的训练数据增大到了7.5M(相较于TexTeller1.0**增加了~15倍**并且数据质量也有所改善)。训练后的TexTeller2.0在测试集中展现出了**更加优越的性能**，尤其在生僻符号、复杂多行、矩阵的识别场景中。
    > 在[这里](./test.pdf)有更多的测试图片以及各家识别模型的横向对比。

## 🔑 前置条件

python=3.10

[pytorch](https://pytorch.org/get-started/locally/)

> [!WARNING]
> 只有CUDA版本>= 12.0被完全测试过，所以最好使用>= 12.0的CUDA版本

## 🖼 关于把latex渲染成图片

* **安装XeLaTex** 并确保`xelatex`可以直接被命令行调用。

* 为了确保正确渲染预测出的公式, 需要在`.tex`文件中**引入以下宏包**:

    ```tex
    \usepackage{multirow,multicol,amsmath,amsfonts,amssymb,mathtools,bm,mathrsfs,wasysym,amsbsy,upgreek,mathalfa,stmaryrd,mathrsfs,dsfont,amsthm,amsmath,multirow}
    ```

## 🚀 开搞

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

> [!NOTE]
> 第一次运行时会在hugging face上下载所需要的checkpoints

## ❓ 常见问题：无法连接到Hugging Face

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

## 🌐 网页演示

首先**确保[poppler](https://poppler.freedesktop.org/)被正确安装，并添加到`PATH`路径中**（终端可以直接使用`pdftoppm`命令）。

然后进入 `TexTeller/src` 目录，运行以下命令

```bash
./start_web.sh
```

在浏览器里输入`http://localhost:8501`就可以看到web demo

> [!TIP]
> 你可以改变`start_web.sh`的默认配置， 例如使用GPU进行推理(e.g. `USE_CUDA=True`) 或者增加beams的数量(e.g. `NUM_BEAM=3`)来获得更高的精确度

> [!IMPORTANT]
> 如果你想直接把预测结果在网页上渲染成图片（比如为了检查预测结果是否正确）你需要确保[xelatex被正确安装](https://github.com/OleehyO/TexTeller?tab=readme-ov-file#-关于把latex渲染成图片)

## 📡 API调用

我们使用[ray serve](https://github.com/ray-project/ray)来对外提供一个TexTeller的API接口，通过使用这个接口，你可以把TexTeller整合到自己的项目里。要想启动server，你需要先进入`TexTeller/src`目录然后运行以下命令:

```bash
python server.py  # default settings
```

你可以给`server.py`传递以下参数来改变server的推理设置(e.g. `python server.py --use_gpu` 来启动GPU推理):

| 参数 | 描述 |
| --- | --- |
| `-ckpt` | 权重文件的路径，*默认为TexTeller的预训练权重*。|
| `-tknz` | 分词器的路径， *默认为TexTeller的分词器*。|
| `-port` | 服务器的服务端口， *默认是8000*。 |
| `--use_gpu` | 是否使用GPU推理，*默认为CPU*。 |
| `--num_beams` | beam search的beam数量， *默认是1*。 |
| `--num_replicas` | 在服务器上运行的服务副本数量， *默认1个副本*。你可以使用更多的副本来获取更大的吞吐量。|
| `--ncpu_per_replica` | 每个服务副本所用的CPU核心数，*默认为1*。 |
| `--ngpu_per_replica` | 每个服务副本所用的GPU数量，*默认为1*。你可以把这个值设置成 0~1之间的数，这样会在一个GPU上运行多个服务副本来共享GPU，从而提高GPU的利用率。(注意，如果 --num_replicas 2, --ngpu_per_replica 0.7, 那么就必须要有2个GPU可用) |

> [!NOTE]
> 一个客户端demo可以在`TexTeller/client/demo.py`找到，你可以参考`demo.py`来给server发送请求

## 🏋️‍♂️ 训练

### 数据集

我们在`TexTeller/src/models/ocr_model/train/dataset`目录中提供了一个数据集的例子，你可以把自己的图片放在`images`目录然后在`formulas.jsonl`中为每张图片标注对应的公式。

准备好数据集后，你需要在`.../dataset/loader.py`中把 **`DIR_URL`变量改成你自己数据集的路径**

### 重新训练分词器

如果你使用了不一样的数据集，你可能需要重新训练tokenizer来得到一个不一样的字典。配置好数据集后，可以通过以下命令来训练自己的tokenizer：

1. 在`TexTeller/src/models/tokenizer/train.py`中，修改`new_tokenizer.save_pretrained('./your_dir_name')`为你自定义的输出目录
    > 注意：如果要用一个不一样大小的字典(默认1W个token)，你需要在 `TexTeller/src/models/globals.py`中修改`VOCAB_SIZE`变量

2. **在 `TexTeller/src` 目录下**运行以下命令:

    ```bash
    python -m models.tokenizer.train
    ```

### 训练模型

要想训练模型, 你需要在`TexTeller/src`目录下运行以下命令：

```bash
python -m models.ocr_model.train.train
```

你可以在`TexTeller/src/models/ocr_model/train/train.py`中设置自己的tokenizer和checkpoint路径（请参考`train.py`）。如果你使用了与TexTeller一样的架构和相同的字典，你还可以用自己的数据集来微调TexTeller的默认权重。

在`TexTeller/src/globals.py`和`TexTeller/src/models/ocr_model/train/train_args.py`中，你可以改变模型的架构以及训练的超参数。

> [!NOTE]
> 我们的训练脚本使用了[Hugging Face Transformers](https://github.com/huggingface/transformers)库, 所以你可以参考他们提供的[文档](https://huggingface.co/docs/transformers/v4.32.1/main_classes/trainer#transformers.TrainingArguments)来获取更多训练参数的细节以及配置。

## 🚧 不足

* 不支持扫描图片以及PDF文档识别

* 不支持手写体公式

## 📅 计划

- [x] ~~使用更大的数据集来训练模型(7.5M样本，即将发布)~~

- [ ] 扫描图片识别

- [ ] PDF文档识别 + 中英文场景支持

- [ ] 推理加速

- [ ] ...

## 💖 感谢

Thanks to [LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR) which has brought me a lot of inspiration, and [im2latex-100K](https://zenodo.org/records/56198#.V2px0jXT6eA) which enriches our dataset.

## ⭐️ 观星曲线

[![Stargazers over time](https://starchart.cc/OleehyO/TexTeller.svg?variant=adaptive)](https://starchart.cc/OleehyO/TexTeller)
