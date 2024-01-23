from ....globals import (
    VOCAB_SIZE,
    OCR_IMG_SIZE,
    OCR_IMG_CHANNELS
)

from typing import (
    Tuple
)

from transformers import (
    DeiTConfig,
    DeiTModel,

    RobertaConfig,
    RobertaModel,
    RobertaTokenizerFast,

    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel
)


class TexTeller:
    def __init__(self, encoder_path=None, decoder_path=None, tokenizer_path=None):
        self.tokenizer = self.get_tokenizer(tokenizer_path)

        assert not (encoder_path is None and decoder_path is not None)
        assert not (encoder_path is not None and decoder_path is None)

        if encoder_path is None:
            encoder_config = DeiTConfig(
                img_size=OCR_IMG_SIZE,
                num_channels=OCR_IMG_CHANNELS
            )

            decoder_config = RobertaConfig(
                vocab_size=VOCAB_SIZE,
                is_decoder=True
            )

            model_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
                encoder_config,
                decoder_config
            )
            self.model = VisionEncoderDecoderModel(model_config)

        else:
            self.model = VisionEncoderDecoderModel.from_pretrained(
                encoder_path,
                decoder_path
            )

        
        ...

    @classmethod
    def get_tokenizer(tokenizer_path: str = None) -> RobertaTokenizerFast:
        if tokenizer_path is None:
            return RobertaTokenizerFast()
        else:
            return RobertaTokenizerFast.from_pretrained(tokenizer_path)
        