import os
import datasets

from pathlib import Path
from transformers import (
    ResNetConfig,
    TrainingArguments,
    Trainer
)

from ..utils import preprocess_fn
from ..model.Resizer import Resizer
from ...globals import NUM_CHANNELS, NUM_CLASSES, RESIZER_IMG_SIZE


def train():
    cur_dirpath = os.getcwd()
    script_dirpath = Path(__file__).resolve().parent
    os.chdir(script_dirpath)

    data = datasets.load_dataset("./dataset").shuffle(seed=42)
    data = data.rename_column("images", "pixel_values")
    data.flatten_indices()
    data = data.with_transform(preprocess_fn)
    train_data, test_data = data['train'], data['test']

    config = ResNetConfig(
        num_channels=NUM_CHANNELS,
        num_labels=NUM_CLASSES,
        img_size=RESIZER_IMG_SIZE
    )
    model = Resizer(config)
    model = Resizer.from_pretrained("/home/lhy/code/TeXify/src/models/resizer/train/train_result_pred_height_v4/checkpoint-213000")

    training_args = TrainingArguments(
        # resume_from_checkpoint="/home/lhy/code/TeXify/src/models/resizer/train/train_result_pred_height_v3/checkpoint-94500",
        max_grad_norm=1.0,
        # use_cpu=True,
        seed=42,                            # 随机种子，用于确保实验的可重复性
        # data_seed=42,                     # data sampler的采样也固定
        # full_determinism=True,            # 使整个训练完全固定（这个设置会有害于模型训练，只用于debug）

        output_dir='./train_result_pred_height_v5',        # 输出目录
        overwrite_output_dir=False,         # 如果输出目录存在，不删除原先的内容
        report_to=["tensorboard"],          # 输出日志到TensorBoard，
                                            #+通过在命令行：tensorboard --logdir ./logs 来查看日志

        logging_dir=None,               # TensorBoard日志文件的存储目录
        log_level="info",
        logging_strategy="steps",           # 每隔一定步数记录一次日志
        logging_steps=500,                  # 记录日志的步数间隔
        logging_nan_inf_filter=False,       # 对loss=nan或inf进行记录

        num_train_epochs=50,                 # 总的训练轮数
        # max_steps=3,                      # 训练的最大步骤数。如果设置了这个参数，
                                            #+那么num_train_epochs将被忽略（通常用于调试）

        # label_names = ['your_label_name'],    # 指定data_loader中的标签名，如果不指定则默认为'labels'

        per_device_train_batch_size=55,     # 每个GPU的batch size
        per_device_eval_batch_size=48*2,      # 每个GPU的evaluation batch size
        auto_find_batch_size=False,         # 自动搜索合适的batch size（指数decay）

        optim = 'adamw_torch',              # 还提供了很多AdamW的变体（相较于经典的AdamW更加高效）
                                            #+当设置了optim后，就不需要在Trainer中传入optimizer
        lr_scheduler_type="cosine",         # 设置lr_scheduler
        warmup_ratio=0.1,                   # warmup占整个训练steps的比例
        # warmup_steps=500,                 # 预热步数
        weight_decay=0,                     # 权重衰减
        learning_rate=5e-5,                 # 学习率
        fp16=False,                         # 是否使用16位浮点数进行训练
        gradient_accumulation_steps=1,      # 梯度累积步数，当batch size无法开很大时，可以考虑这个参数来实现大batch size的效果
        gradient_checkpointing=False,       # 当为True时，会在forward时适当丢弃一些中间量（用于backward），从而减轻显存压力（但会增加forward的时间）
        label_smoothing_factor=0.0,         # softlabel，等于0时表示未开启
        # debug='underflow_overflow',       # 训练时检查溢出，如果发生，则会发出警告。（该模式通常用于debug）
        torch_compile=True,                # 是否使用torch.compile来编译模型（从而获得更好的训练和推理性能）
                                            #+ 要求torch > 2.0，并且这个功能现在还不是很稳定
        # deepspeed='your_json_path',       #  使用deepspeed来训练，需要指定ds_config.json的路径
                                            #+ 在Trainer中使用Deepspeed时一定要注意ds_config.json中的配置是否与Trainer的一致（如学习率，batch size，梯度累积步数等）
                                            #+ 如果不一致，会出现很奇怪的bug（而且一般还很难发现）													

        dataloader_pin_memory=True,         # 可以加快数据在cpu和gpu之间转移的速度
        dataloader_num_workers=16,           # 默认不会使用多进程来加载数据
        dataloader_drop_last=True,          # 丢掉最后一个minibatch

        evaluation_strategy="steps",        # 评估策略，可以是"steps"或"epoch"
        eval_steps=500,                       # if evaluation_strategy="step"
        # eval_steps=10,                     # if evaluation_strategy="step"

        save_strategy="steps",              # 保存checkpoint的策略
        save_steps=1500,                    # 模型保存的步数间隔
        save_total_limit=5,                 # 保存的模型的最大数量。如果超过这个数量，最旧的模型将被删除

        load_best_model_at_end=True,        # 训练结束时是否加载最佳模型
        metric_for_best_model="eval_loss",  # 用于选择最佳模型的指标
        greater_is_better=False,            # 指标值越小越好

        do_train=True,                      # 是否进行训练，通常用于调试
        do_eval=True,                       # 是否进行评估，通常用于调试

        remove_unused_columns=True,         # 是否删除没有用到的列（特征），默认为True
                                            #+当删除了没用到的列后，making it easier to unpack inputs into the model’s call function

        push_to_hub=False,                  # 是否训练完后上传hub，需要先在命令行：huggingface-cli login进行登录认证的配置，配置完后，认证信息会存到cache文件夹里
        hub_model_id="a_different_name",    # 模型的名字
                                            #+每次保存模型时，都会上传到hub，
                                            #+训练完后，记得trainer.push_to_hub()，会将模型使用的参数以及验证集上的结果传到hub上 
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
    )
    trainer.train()

    os.chdir(cur_dirpath)


if __name__ == '__main__':
    train()
