from PIL import Image
from pathlib import Path

from models.globals import (
    VOCAB_SIZE,
    OCR_IMG_SIZE,
    OCR_IMG_CHANNELS,
    MAX_TOKEN_SIZE
)

from transformers import (
    ViTConfig,
    ViTModel,
    TrOCRConfig,
    TrOCRForCausalLM,
    RobertaTokenizerFast,
    VisionEncoderDecoderModel,
)


class TexTeller(VisionEncoderDecoderModel):
    def __init__(self, decoder_path=None, tokenizer_path=None):
        encoder = ViTModel(ViTConfig(
            image_size=OCR_IMG_SIZE,
            num_channels=OCR_IMG_CHANNELS
        ))
        decoder = TrOCRForCausalLM(TrOCRConfig(
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=MAX_TOKEN_SIZE
        ))
        super().__init__(encoder=encoder, decoder=decoder)
    
    @classmethod
    def from_pretrained(cls, model_path: str):
        model_path = Path(model_path).resolve()
        return VisionEncoderDecoderModel.from_pretrained(str(model_path))

    @classmethod
    def get_tokenizer(cls, tokenizer_path: str) -> RobertaTokenizerFast:
        tokenizer_path = Path(tokenizer_path).resolve()
        return RobertaTokenizerFast.from_pretrained(str(tokenizer_path))


if __name__ == "__main__":
    # texteller = TexTeller()
    from ..utils.inference import inference
    model = TexTeller.from_pretrained('/home/lhy/code/TeXify/src/models/ocr_model/train/train_result/checkpoint-57500')
    tokenizer = TexTeller.get_tokenizer('/home/lhy/code/TeXify/src/models/tokenizer/roberta-tokenizer-550Kformulas')

    base = '/home/lhy/code/TeXify/src/models/ocr_model/model'
    imgs_path = [
        # base + '/1.jpg',
        # base + '/2.jpg',
        # base + '/3.jpg',
        # base + '/4.jpg',
        # base + '/5.jpg',
        # base + '/6.jpg',
        base + '/foo.jpg'
    ]

    # res = inference(model, [img1, img2, img3, img4, img5, img6, img7], tokenizer)
    res = inference(model, imgs_path, tokenizer)
    pause = 1

