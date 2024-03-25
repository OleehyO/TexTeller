from PIL import Image
from pathlib import Path
import datasets
import json

DIR_URL = Path('absolute/path/to/dataset/directory')
# e.g. DIR_URL = Path('/home/OleehyO/TeXTeller/src/models/ocr_model/train/dataset')


class LatexFormulas(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = []

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                "image": datasets.Image(),
                "latex_formula": datasets.Value("string")
            })
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        dir_path = Path(dl_manager.download(str(DIR_URL)))
        assert dir_path.is_dir()

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'dir_path': dir_path,
                }
            )
        ]

    def _generate_examples(self, dir_path: Path):
        images_path   = dir_path / 'images'
        formulas_path = dir_path / 'formulas.jsonl'

        img2formula = {}
        with formulas_path.open('r', encoding='utf-8') as f:
            for line in f:
                single_json = json.loads(line)
                img2formula[single_json['img_name']] = single_json['formula']

        for img_path in images_path.iterdir():
            if img_path.suffix not in ['.jpg', '.png']:
                continue
            yield str(img_path), {
                "image": Image.open(img_path),
                "latex_formula": img2formula[img_path.name]
            }
