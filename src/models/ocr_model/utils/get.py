from ....globals import VOCAB_SIZE
from typing import (
    Tuple
)

from transformers import (
    RobertaConfig,
    RobertaModel,
    RobertaTokenizerFast
)


def get_encoder():
    ...


def get_tokenizer() -> RobertaTokenizerFast:
    ...


def get_decoder() -> RobertaModel:
    configuration = RobertaConfig(
        vocab_size=VOCAB_SIZE,
        is_decoder=True
    )
    model = RobertaModel(configuration)
    return model

