CONFIG = {
    "seed": 42,                            # Random seed for reproducibility
    "use_cpu": False,                      # Whether to use CPU (it's easier to debug with CPU when starting to test the code)
    "learning_rate": 5e-5,                 # Learning rate
    "num_train_epochs": 10,                # Total number of training epochs
    "per_device_train_batch_size": 4,      # Batch size per GPU for training
    "per_device_eval_batch_size": 8,       # Batch size per GPU for evaluation

    "output_dir": "train_result",          # Output directory
    "overwrite_output_dir": False,         # If the output directory exists, do not delete its content
    "report_to": ["tensorboard"],          # Report logs to TensorBoard

    "save_strategy": "steps",              # Strategy to save checkpoints
    "save_steps": 500,                     # Interval of steps to save checkpoints, can be int or a float (0~1), when float it represents the ratio of total training steps (e.g., can set to 1.0 / 2000)
    "save_total_limit": 5,                 # Maximum number of models to save. The oldest models will be deleted if this number is exceeded

    "logging_strategy": "steps",           # Log every certain number of steps
    "logging_steps": 500,                  # Number of steps between each log
    "logging_nan_inf_filter": False,       # Record logs for loss=nan or inf

    "optim": "adamw_torch",                # Optimizer
    "lr_scheduler_type": "cosine",         # Learning rate scheduler
    "warmup_ratio": 0.1,                   # Ratio of warmup steps in total training steps (e.g., for 1000 steps, the first 100 steps gradually increase lr from 0 to the set lr)
    "max_grad_norm": 1.0,                  # For gradient clipping, ensure the norm of the gradients does not exceed 1.0 (default 1.0)
    "fp16": False,                         # Whether to use 16-bit floating point for training (generally not recommended, as loss can easily explode)
    "bf16": False,                         # Whether to use Brain Floating Point (bfloat16) for training (recommended if architecture supports it)
    "gradient_accumulation_steps": 1,      # Gradient accumulation steps, consider this parameter to achieve large batch size effects when batch size cannot be large
    "jit_mode_eval": False,                # Whether to use PyTorch jit trace during eval (can speed up the model, but the model must be static, otherwise will throw errors)
    "torch_compile": False,                # Whether to use torch.compile to compile the model (for better training and inference performance)

    "dataloader_pin_memory": True,         # Can speed up data transfer between CPU and GPU
    "dataloader_num_workers": 1,           # Default is not to use multiprocessing for data loading, usually set to 4*number of GPUs used

    "evaluation_strategy": "steps",        # Evaluation strategy, can be "steps" or "epoch"
    "eval_steps": 500,                     # If evaluation_strategy="step"

    "remove_unused_columns": False,        # Don't change this unless you really know what you are doing.
}
