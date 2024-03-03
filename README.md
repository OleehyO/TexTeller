<div align="center">
<h1><img src="./assets/fire.svg" width=30, height=30> 
𝚃𝚎𝚡𝚃𝚎𝚕𝚕𝚎𝚛 <img src="./assets/fire.svg" width=30, height=30> </h1>

<p align="center">
English | <a href="./assets/README_zh.md">中文版本</a>
</p>

<p align="center">
  <img src="./assets/web_demo.gif" alt="TexTeller_demo" width=800>
</p>

</div>

TexTeller is a ViT-based model designed for end-to-end formula recognition. It can recognize formulas in natural images and convert them into LaTeX-style formulas.

TexTeller is trained on a larger dataset of image-formula pairs (a 550K dataset available [here](https://huggingface.co/datasets/OleehyO/latex-formulas)), **exhibits superior generalization ability and higher accuracy compared to [LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR)**, which uses approximately 100K data points. This larger dataset enables TexTeller to cover most usage scenarios more effectively( **excluding scanned images and handwritten formulas** ).
> A TexTeller checkpoint trained on a 5.5M dataset will be released soon.

## Prerequisites

python=3.10

pytorch

> Note: Only CUDA version >= 12.0 have been fully tested, so we recommend using CUDA version>=12.0

## Getting Started

1. Clone the repository:

    ```bash
    git clone https://github.com/OleehyO/TexTeller
    ```

2. After [pytorch installation](https://pytorch.org/get-started/locally/#start-locally), install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Navigate to the `TexTeller/src` directory and run the following command to perform inference:

    ```bash
    python inference.py -img "/path/to/image.{jpg,png}" 
    # use -cuda option to enable GPU inference
    #+e.g. python inference.py -img "./img.jpg" -cuda
    ```

    > Checkpoints will be downloaded in your first run.

## Web Demo

You can also run the web demo by navigating to the `TexTeller/src` directory and running the following command:

```bash
./start_web.sh
```

Then go to `http://localhost:8501` in your browser to run TexTeller in the web.

> You can change the default settings in `start_web.sh`, such as inference with GPU(e.g. `USE_CUDA=True`) or increase the number of beams(e.g. `NUM_BEAM=3`) for higher accuracy.

## API

We use [ray serve](https://github.com/ray-project/ray) to provide a simple API for using TexTeller in your own projects. To start the server, navigate to the `TexTeller/src` directory and run the following command:

```bash
python server.py  # default settings
```

You can pass the following arguments to the `server.py` script to get custom inference settings(e.g. `python server.py --use_gpu` to enable GPU inference):

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

> Client demo can be found in `TexTeller/client/demo.py`, you can refer to `demo.py` to send requests to the server.

## Training

### Dataset

We provide a dataset example in `TexTeller/src/models/ocr_model/train/dataset`, and you can place your own images in the `images` directory and annotate the corresponding formula for each image in `formulas.jsonl`.

After the dataset is ready, you should **change the `DIR_URL` variable** in `.../dataset/loader.py` to the path of your dataset.

### Retrain the tokenizer

If you are using a different dataset, you may need to retrain the tokenizer to match your specific vocabulary. After setting up the dataset, you can do this by:

1. Change the line `new_tokenizer.save_pretrained('./your_dir_name')` in `TexTeller/src/models/tokenizer/train.py` to your desired output directory name.
    > To use a different vocabulary size, you should modify the `VOCAB_SIZE` parameter in the `TexTeller/src/models/globals.py`.

2. Running the following command **under `TexTeller/src` directory**:

    ```bash
    python -m models.tokenizer.train
    ```

### Train the model

To train the model, you can run the following command **under `TexTeller/src` directory**:

```bash
python -m models.ocr_model.train.train
```

You can set your own tokenizer and checkpoint path(or fine-tune the default model checkpoint if you don't use your own tokenizer while keeping the same model architecture) in `TexTeller/src/models/ocr_model/train/train.py`.
> Please refer to `train.py` for more details.

Model architecture and training hyperparameters can be adjusted in `TexTeller/src/globals.py` and `TexTeller/src/models/ocr_model/train/train_args.py`.

> We use the [Hugging Face Transformers](https://github.com/huggingface/transformers) library for model training, so you can find more details about the training hyperparameters in their [documentation](https://huggingface.co/docs/transformers/v4.32.1/main_classes/trainer#transformers.TrainingArguments).

## To-Do

- [ ] Train our model with a larger amount of data(5.5M samples, and soon to be released).

- [ ] Inference acceleration.

- [ ] ...

## Acknowledgements

Thanks to [LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR) which has brought me a lot of inspiration, and [im2latex-100K](https://zenodo.org/records/56198#.V2px0jXT6eA) which enriches our dataset.
