# 公式图片(灰度化后)的均值和方差
IMAGE_MEAN = 0.9545467
IMAGE_STD  = 0.15394445


# =========================   ocr模型用的参数   ============================= #

# 输入图片的最大最小的宽和高
MIN_HEIGHT = 32
MAX_HEIGHT = 512
MIN_WIDTH  = 32
MAX_WIDTH  = 1280
# LaTex-OCR中分别是 32、192、32、672

# ocr模型所用数据集，pdf转图片所用的Density值(dpi)
TEXIFY_INPUT_DENSITY = 100

# ocr模型的tokenizer中的词典数量
VOCAB_SIZE = 10000

# ocr模型是否固定输入图片的大小
OCR_FIX_SIZE = True
# ocr模型训练时，输入图片所固定的大小 (when OCR_FIX_SIZE is True)
OCR_IMG_SIZE = 448
# ocr模型训练时，输入图片最大的宽和高（when OCR_FIX_SIZE is False）
OCR_IMG_MAX_HEIGHT = 512
OCR_IMG_MAX_WIDTH  = 768

# ocr模型输入图片的通道数
OCR_IMG_CHANNELS = 1  # 灰度图

# ocr模型训练数据集的最长token数
MAX_TOKEN_SIZE = 512     # 模型最长的embedding长度被设置成了512，所以这里必须是512
# MAX_TOKEN_SIZE = 600

# ocr模型训练时随机缩放的比例
MAX_RESIZE_RATIO = 1.15
MIN_RESIZE_RATIO = 0.75

# ocr模型输入的图片要求的最低宽和高(过滤垃圾数据)
MIN_HEIGHT = 12
MIN_WIDTH  = 30

# ============================================================================= #


# =========================   Resizer模型用的参数   ============================= #

# Resizer模型所用数据集中，图片所用的Density渲染值
RESIZER_INPUT_DENSITY  = 200   

LABEL_RATIO   = 1.0 * TEXIFY_INPUT_DENSITY / RESIZER_INPUT_DENSITY

NUM_CLASSES   = 1      # 模型使用回归预测
NUM_CHANNELS  = 1      # 输入单通道图片（灰度图）

# Resizer在训练时，图片所固定的的大小
RESIZER_IMG_SIZE = 448    
# ============================================================================= #
