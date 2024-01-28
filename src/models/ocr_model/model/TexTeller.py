from ....globals import (
    VOCAB_SIZE,
    OCR_IMG_SIZE,
    OCR_IMG_CHANNELS
)

from transformers import (
    ViTConfig,
    ViTModel,

    TrOCRConfig,
    TrOCRForCausalLM,

    RobertaTokenizerFast,

    VisionEncoderDecoderModel
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
    texteller = TexTeller()
    tokenizer = texteller.get_tokenizer('/home/lhy/code/TeXify/src/models/tokenizer/roberta-tokenizer-550Kformulas')
    foo = ["Hello, my name is LHY.", "I am a researcher at the University of Science and Technology of China."]
    bar = tokenizer(foo, return_special_tokens_mask=True)
    pause = 1

