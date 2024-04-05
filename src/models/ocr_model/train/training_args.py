CONFIG = {
    "seed": 42,                            # 随机种子，用于确保实验的可重复性
    "use_cpu": False,                      # 是否使用cpu（刚开始测试代码的时候先用cpu跑会更容易debug）
    # "data_seed": 42,                     # data sampler的采样也固定
    # "full_determinism": True,            # 使整个训练完全固定（这个设置会有害于模型训练，只用于debug）

    "output_dir": "train_result/TexTellerv3",          # 输出目录
    "overwrite_output_dir": False,         # 如果输出目录存在，不删除原先的内容
    "report_to": ["tensorboard"],          # 输出日志到TensorBoard，
                                           #+通过在命令行：tensorboard --logdir ./logs 来查看日志

    "logging_dir": None,                   # TensorBoard日志文件的存储目录(使用默认值)
    "log_level": "warning",                   # 其他可选:‘debug’, ‘info’, ‘warning’, ‘error’ and ‘critical’（由低级别到高级别）
    "logging_strategy": "steps",           # 每隔一定步数记录一次日志
    "logging_steps": 4000,                  # 记录日志的步数间隔，可以是int也可以是(0~1)的float，当是float时表示总的训练步数的ratio(比方说可以设置成1.0 / 2000)
                                           #+通常与eval_steps一致
    "logging_nan_inf_filter": False,       # 对loss=nan或inf进行记录

    "num_train_epochs": 4,                # 总的训练轮数
    # "max_steps": 3,                      # 训练的最大步骤数。如果设置了这个参数，
                                           #+那么num_train_epochs将被忽略（通常用于调试）

    # "label_names": ['your_label_name'],  # 指定data_loader中的标签名，如果不指定则默认为'labels'

    "per_device_train_batch_size": 3,    # 每个GPU的batch size
    "per_device_eval_batch_size": 6,      # 每个GPU的evaluation batch size
    # "auto_find_batch_size": True,          # 自动搜索合适的batch size（指数decay）
    "auto_find_batch_size": False,          # 自动搜索合适的batch size（指数decay）

    "optim": "adamw_torch",                # 还提供了很多AdamW的变体（相较于经典的AdamW更加高效）
                                           #+当设置了optim后，就不需要在Trainer中传入optimizer
    "lr_scheduler_type": "cosine",         # 设置lr_scheduler
    "warmup_ratio": 0.1,                   # warmup占整个训练steps的比例(假如训练1000步，那么前100步就是从lr=0慢慢长到参数设定的lr)
    # "warmup_steps": 500,                 # 预热步数, 这个参数与warmup_ratio是矛盾的
    "weight_decay": 0,                     # 权重衰减
    "learning_rate": 5e-5,                 # 学习率
    "max_grad_norm": 1.0,                  # 用于梯度裁剪，确保梯度的范数不超过1.0（默认1.0）
    "fp16": False,                         # 是否使用16位浮点数进行训练（一般不推荐，loss很容易炸）
    "bf16": False,                         # 是否使用16位宽浮点数进行训练（如果架构支持的话推荐使用）
    "gradient_accumulation_steps": 2,      # 梯度累积步数，当batch size无法开很大时，可以考虑这个参数来实现大batch size的效果
    "gradient_checkpointing": False,       # 当为True时，会在forward时适当丢弃一些中间量（用于backward），从而减轻显存压力（但会增加forward的时间）
    "label_smoothing_factor": 0.0,         # softlabel，等于0时表示未开启
    # "debug": "underflow_overflow",       # 训练时检查溢出，如果发生，则会发出警告。（该模式通常用于debug）
    "jit_mode_eval": True,                 # 是否在eval的时候使用PyTorch jit trace（可以加速模型，但模型必须是静态的，否则会报错）
    "torch_compile": True,                 # 是否使用torch.compile来编译模型（从而获得更好的训练和推理性能）
                                           #+ 要求torch > 2.0，这个功能很好使，当模型跑通的时候可以开起来
    # "deepspeed": "your_json_path",       #  使用deepspeed来训练，需要指定ds_config.json的路径
                                           #+ 在Trainer中使用Deepspeed时一定要注意ds_config.json中的配置是否与Trainer的一致（如学习率，batch size，梯度累积步数等）
                                           #+ 如果不一致，会出现很奇怪的bug（而且一般还很难发现）													

    "dataloader_pin_memory": True,         # 可以加快数据在cpu和gpu之间转移的速度
    "dataloader_num_workers": 16,          # 默认不会使用多进程来加载数据，通常设成4*所用的显卡数
    "dataloader_drop_last": True,          # 丢掉最后一个minibatch，保证训练的梯度稳定

    "evaluation_strategy": "steps",        # 评估策略，可以是"steps"或"epoch"
    "eval_steps": 4000,                     # if evaluation_strategy="step"
                                           #+默认情况下与logging_steps一样，可以是int也可以是(0~1)的float，当是float时表示总的训练步数的ratio(比方说可以设置成1.0 / 2000)

    "save_strategy": "steps",              # 保存checkpoint的策略
    "save_steps": 4000,                     # checkpoint保存的步数间隔，可以是int也可以是(0~1)的float，当是float时表示总的训练步数的ratio(比方说可以设置成1.0 / 2000)
    "save_total_limit": 10,                 # 保存的模型的最大数量。如果超过这个数量，最旧的模型将被删除

    "load_best_model_at_end": True,        # 训练结束时是否加载最佳模型
                                           #+当设置True时，会保存训练时评估结果最好的checkpoint
                                           #+当设置True时，evaluation_strategy必须与save_strategy一样，并且save_steps必须是eval_steps的整数倍
    "metric_for_best_model": "eval_loss",  # 用于选择最佳模型的指标(必须与load_best_model_at_end一起用)
                                           #+可以使用compute_metrics输出的evaluation的结果中（一个字典）的某个值
                                           #+注意：Trainer会在compute_metrics输出的字典的键前面加上一个prefix，默认就是“eval_”
    "greater_is_better": False,            # 指标值越小越好(必须与metric_for_best_model一起用)

    "do_train": True,                      # 是否进行训练，通常用于调试
    "do_eval": True,                       # 是否进行评估，通常用于调试

    "remove_unused_columns": False,        # 是否删除没有用到的列（特征），默认为True
                                           #+当删除了没用到的列后，making it easier to unpack inputs into the model’s call function
    #+注意：remove_unused_columns去除列的操作会把传入的dataset的columns_names与模型forward方法中的参数名进行配对，对于不存在forward方法中的列名就会直接删掉整个feature
    #+因此如果在dataset.with_transform(..)中给数据进行改名，那么这个remove操作会直接把原始的数据直接删掉，从而导致之后会拿到一个空的dataset，导致在对dataset进行切片取值时出问题
    #+例如读进来的dataset图片对应的feature name叫"images"，而模型forward方法中对应的参数名叫“pixel_values”，
    #+此时如果是在data.withtransfrom(..)中根据这个"images"生成其他模型forward方法中需要的参数，然后再把"images"改名成“pixel_values”，那么整个过程就会出问题
    #+因为设置了remove_unused_columns=True后，会先给dataset进行列名检查，然后“images”这个feature会直接被删掉（导致with_transform的transform_fn拿不到“images”这个feature）
    #+所以一个good practice就是：对于要改名的特征，先提前使用dataset.rename_column进行改名

    "push_to_hub": False,                  # 是否训练完后上传hub，需要先在命令行：huggingface-cli login进行登录认证的配置，配置完后，认证信息会存到cache文件夹里
}