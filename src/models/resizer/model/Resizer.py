from transformers import ResNetForImageClassification

class Resizer(ResNetForImageClassification):
    def __init__(self, config):
        super().__init__(config)
