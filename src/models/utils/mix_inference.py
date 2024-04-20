import re
import heapq
import cv2
import numpy as np

from collections import Counter
from typing import List
from PIL import Image

from surya.detection import batch_text_detection
from surya.input.processing import slice_polys_from_image
from surya.recognition import batch_recognition

from ..det_model.inference import predict as latex_det_predict
from ..det_model.Bbox import Bbox, draw_bboxes

from ..ocr_model.utils.inference import inference as latex_rec_predict
from ..ocr_model.utils.to_katex import to_katex

MAXV = 999999999


def mask_img(img, bboxes: List[Bbox], bg_color: np.ndarray) -> np.ndarray:
    mask_img = img.copy()
    for bbox in bboxes:
        mask_img[bbox.p.y:bbox.p.y + bbox.h, bbox.p.x:bbox.p.x + bbox.w] = bg_color
    return mask_img


def bbox_merge(sorted_bboxes: List[Bbox]) -> List[Bbox]:
    if (len(sorted_bboxes) == 0):
        return []
    bboxes = sorted_bboxes.copy()
    guard = Bbox(MAXV, bboxes[-1].p.y, -1, -1, label="guard")
    bboxes.append(guard)
    res = []
    prev = bboxes[0]
    for curr in bboxes:
        if prev.ur_point.x <= curr.p.x or not prev.same_row(curr):
            res.append(prev)
            prev = curr
        else:
            prev.w = max(prev.w, curr.ur_point.x - prev.p.x)
    return res


def split_conflict(ocr_bboxes: List[Bbox], latex_bboxes: List[Bbox]) -> List[Bbox]:
    if latex_bboxes == []:
        return ocr_bboxes
    if ocr_bboxes == [] or len(ocr_bboxes) == 1:
        return ocr_bboxes

    bboxes = sorted(ocr_bboxes + latex_bboxes)

    # log results
    for idx, bbox in enumerate(bboxes):
        bbox.content = str(idx)
    draw_bboxes(Image.fromarray(img), bboxes, name="before_split_confict.png")

    assert len(bboxes) > 1

    heapq.heapify(bboxes)
    res = []
    candidate = heapq.heappop(bboxes)
    curr = heapq.heappop(bboxes)
    idx = 0
    while (len(bboxes) > 0):
        idx += 1
        assert candidate.p.x < curr.p.x or not candidate.same_row(curr)

        if candidate.ur_point.x <= curr.p.x or not candidate.same_row(curr):
            res.append(candidate)
            candidate = curr
            curr = heapq.heappop(bboxes)
        elif candidate.ur_point.x < curr.ur_point.x:
            assert not (candidate.label != "text" and curr.label != "text")
            if candidate.label == "text" and curr.label == "text":
                candidate.w = curr.ur_point.x - candidate.p.x
                curr = heapq.heappop(bboxes)
            elif candidate.label != curr.label:
                if candidate.label == "text":
                    candidate.w = curr.p.x - candidate.p.x
                    res.append(candidate)
                    candidate = curr
                    curr = heapq.heappop(bboxes)
                else:
                    curr.w = curr.ur_point.x - candidate.ur_point.x
                    curr.p.x = candidate.ur_point.x
                    heapq.heappush(bboxes, curr)
                    curr = heapq.heappop(bboxes)
                
        elif candidate.ur_point.x >= curr.ur_point.x:
            assert not (candidate.label != "text" and curr.label != "text")

            if candidate.label == "text":
                assert curr.label != "text"
                heapq.heappush(
                    bboxes,
                    Bbox(
                        curr.ur_point.x,
                        candidate.p.y,
                        candidate.h,
                        candidate.ur_point.x - curr.ur_point.x,
                        label="text",
                        confidence=candidate.confidence,
                        content=None
                    )
                )
                candidate.w = curr.p.x - candidate.p.x
                res.append(candidate)
                candidate = curr
                curr = heapq.heappop(bboxes)
            else:
                assert curr.label == "text"
                curr = heapq.heappop(bboxes)
        else:
            assert False
    res.append(candidate)
    res.append(curr)

    # log results
    for idx, bbox in enumerate(res):
        bbox.content = str(idx)
    draw_bboxes(Image.fromarray(img), res, name="after_split_confict.png")

    return res


def mix_inference(
    img_path: str,
    language: str,
    infer_config,
    latex_det_model,

    lang_ocr_models,

    latex_rec_models,
    accelerator="cpu",
    num_beams=1
) -> str:
    '''
    Input a mixed image of formula text and output str (in markdown syntax)
    '''
    global img
    img = cv2.imread(img_path)
    corners = [tuple(img[0, 0]), tuple(img[0, -1]),
               tuple(img[-1, 0]), tuple(img[-1, -1])]
    bg_color = np.array(Counter(corners).most_common(1)[0][0])

    latex_bboxes = latex_det_predict(img_path, latex_det_model, infer_config)
    latex_bboxes = sorted(latex_bboxes)
    # log results
    draw_bboxes(Image.fromarray(img), latex_bboxes, name="latex_bboxes(unmerged).png")
    latex_bboxes = bbox_merge(latex_bboxes)
    # log results
    draw_bboxes(Image.fromarray(img), latex_bboxes, name="latex_bboxes(merged).png")
    masked_img = mask_img(img, latex_bboxes, bg_color)

    det_model, det_processor, rec_model, rec_processor = lang_ocr_models
    images = [Image.fromarray(masked_img)]
    det_prediction = batch_text_detection(images, det_model, det_processor)[0]
    # log results
    draw_bboxes(Image.fromarray(img), latex_bboxes, name="ocr_bboxes(unmerged).png")

    lang = [language]
    slice_map = []
    all_slices = []
    all_langs = []
    ocr_bboxes = [
        Bbox(
            p.bbox[0], p.bbox[1], p.bbox[3] - p.bbox[1], p.bbox[2] - p.bbox[0],
            label="text",
            confidence=p.confidence,
            content=None
        )
        for p in det_prediction.bboxes
    ]
    ocr_bboxes = sorted(ocr_bboxes)
    ocr_bboxes = bbox_merge(ocr_bboxes)
    draw_bboxes(Image.fromarray(img), ocr_bboxes, name="ocr_bboxes(merged).png")
    ocr_bboxes = split_conflict(ocr_bboxes, latex_bboxes)
    ocr_bboxes = list(filter(lambda x: x.label == "text", ocr_bboxes))
    polygons = [
        [
            [bbox.ul_point.x, bbox.ul_point.y],
            [bbox.ur_point.x, bbox.ur_point.y],
            [bbox.lr_point.x, bbox.lr_point.y],
            [bbox.ll_point.x, bbox.ll_point.y]
        ]
        for bbox in ocr_bboxes
    ]

    slices = slice_polys_from_image(images[0], polygons)
    slice_map.append(len(slices))
    all_slices.extend(slices)
    all_langs.extend([lang] * len(slices))

    rec_predictions, _ = batch_recognition(all_slices, all_langs, rec_model, rec_processor)

    assert len(rec_predictions) == len(ocr_bboxes)
    for content, bbox in zip(rec_predictions, ocr_bboxes):
        bbox.content = content
    
    latex_imgs =[]
    for bbox in latex_bboxes:
        latex_imgs.append(img[bbox.p.y:bbox.p.y + bbox.h, bbox.p.x:bbox.p.x + bbox.w])
    latex_rec_res = latex_rec_predict(*latex_rec_models, latex_imgs, accelerator, num_beams, max_tokens=200)
    for bbox, content in zip(latex_bboxes, latex_rec_res):
        bbox.content = to_katex(content)
        if bbox.label == "embedding":
            bbox.content = " $" + bbox.content + "$ "
        elif bbox.label == "isolated":
            bbox.content = '\n' + r"$$" + bbox.content + r"$$" + '\n'

    bboxes = sorted(ocr_bboxes + latex_bboxes)
    if bboxes == []:
        return ""

    md = ""
    prev = Bbox(bboxes[0].p.x, bboxes[0].p.y, -1, -1, label="guard")
    for curr in bboxes:
        if not prev.same_row(curr):
            md += "\n"
        md += curr.content
        if (
            prev.label == "isolated"
            and curr.label == "text"
            and bool(re.fullmatch(r"\([1-9]\d*?\)", curr.content))
        ):
            md += '\n'
        prev = curr
    return md
