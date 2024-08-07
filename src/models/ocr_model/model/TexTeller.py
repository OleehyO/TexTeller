from pathlib import Path

from ...globals import (
    VOCAB_SIZE,
    FIXED_IMG_SIZE,
    IMG_CHANNELS,
    MAX_TOKEN_SIZE
)

from transformers import (
    RobertaTokenizerFast,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig
)


class TexTeller(VisionEncoderDecoderModel):
    REPO_NAME = 'OleehyO/TexTeller'
    def __init__(self):
        config = VisionEncoderDecoderConfig.from_pretrained(Path(__file__).resolve().parent / "config.json")
        config.encoder.image_size              = FIXED_IMG_SIZE
        config.encoder.num_channels            = IMG_CHANNELS
        config.decoder.vocab_size              = VOCAB_SIZE
        config.decoder.max_position_embeddings = MAX_TOKEN_SIZE

        super().__init__(config=config)
    
    @classmethod
    def from_pretrained(cls, model_path: str = None, use_onnx=False, onnx_provider=None):
        if model_path is None or model_path == 'default':
            if not use_onnx:
                return VisionEncoderDecoderModel.from_pretrained(cls.REPO_NAME)
            else:
                from optimum.onnxruntime import ORTModelForVision2Seq
                use_gpu = True if onnx_provider == 'cuda' else False
                return ORTModelForVision2Seq.from_pretrained(cls.REPO_NAME, provider="CUDAExecutionProvider" if use_gpu else "CPUExecutionProvider")
        model_path = Path(model_path).resolve()
        return VisionEncoderDecoderModel.from_pretrained(str(model_path))

    @classmethod
    def get_tokenizer(cls, tokenizer_path: str = None) -> RobertaTokenizerFast:
        if tokenizer_path is None or tokenizer_path == 'default':
            return RobertaTokenizerFast.from_pretrained(cls.REPO_NAME)
        tokenizer_path = Path(tokenizer_path).resolve()
        return RobertaTokenizerFast.from_pretrained(str(tokenizer_path))
