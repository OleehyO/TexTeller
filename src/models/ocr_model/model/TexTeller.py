from PIL import Image

from ....globals import (
    VOCAB_SIZE,
    OCR_IMG_SIZE,
    OCR_IMG_CHANNELS,
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
        ))
        super().__init__(encoder=encoder, decoder=decoder)
    
    @classmethod
    def from_pretrained(cls, model_path: str):
        return VisionEncoderDecoderModel.from_pretrained(model_path)

    @classmethod
    def get_tokenizer(cls, tokenizer_path: str) -> RobertaTokenizerFast:
        return RobertaTokenizerFast.from_pretrained(tokenizer_path)


if __name__ == "__main__":
    # texteller = TexTeller()
    from ..inference import inference
    model = TexTeller.from_pretrained('/home/lhy/code/TeXify/src/models/ocr_model/train/train_result/checkpoint-22500')
    tokenizer = TexTeller.get_tokenizer('/home/lhy/code/TeXify/src/models/tokenizer/roberta-tokenizer-550Kformulas')

    img1 = Image.open('/home/lhy/code/TeXify/src/models/ocr_model/model/1.png')
    img2 = Image.open('/home/lhy/code/TeXify/src/models/ocr_model/model/2.png')
    img3 = Image.open('/home/lhy/code/TeXify/src/models/ocr_model/model/3.png')
    img4 = Image.open('/home/lhy/code/TeXify/src/models/ocr_model/model/4.png')
    img5 = Image.open('/home/lhy/code/TeXify/src/models/ocr_model/model/5.png')
    img6 = Image.open('/home/lhy/code/TeXify/src/models/ocr_model/model/6.png')

    res = inference(model, [img1, img2, img3, img4, img5, img6], tokenizer)
    pause = 1

