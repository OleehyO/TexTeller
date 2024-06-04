import re
import heapq
import cv2
import time
import numpy as np

from collections import Counter
from typing import List
from PIL import Image

from ..det_model.inference import predict as latex_det_predict
from ..det_model.Bbox import Bbox, draw_bboxes

from ..ocr_model.utils.inference import inference as latex_rec_predict
from ..ocr_model.utils.to_katex import to_katex, change_all

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
        assert candidate.p.x <= curr.p.x or not candidate.same_row(curr)

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


def slice_from_image(img: np.ndarray, ocr_bboxes: List[Bbox]) -> List[np.ndarray]:
    sliced_imgs = []
    for bbox in ocr_bboxes:
        x, y = int(bbox.p.x), int(bbox.p.y)
        w, h = int(bbox.w), int(bbox.h)
        sliced_img = img[y:y+h, x:x+w]
        sliced_imgs.append(sliced_img)
    return sliced_imgs


def mix_inference(
    img_path: str,
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

    start_time = time.time()
    latex_bboxes = latex_det_predict(img_path, latex_det_model, infer_config)
    end_time = time.time()
    print(f"latex_det_model time: {end_time - start_time:.2f}s")
    latex_bboxes = sorted(latex_bboxes)
    # log results
    draw_bboxes(Image.fromarray(img), latex_bboxes, name="latex_bboxes(unmerged).png")
    latex_bboxes = bbox_merge(latex_bboxes)
    # log results
    draw_bboxes(Image.fromarray(img), latex_bboxes, name="latex_bboxes(merged).png")
    masked_img = mask_img(img, latex_bboxes, bg_color)

    det_model, rec_model = lang_ocr_models
    start_time = time.time()
    det_prediction, _ = det_model(masked_img)
    end_time = time.time()
    print(f"ocr_det_model time: {end_time - start_time:.2f}s")
    ocr_bboxes = [
        Bbox(
            p[0][0], p[0][1], p[3][1]-p[0][1], p[1][0]-p[0][0],
            label="text",
            confidence=None,
            content=None
        )
        for p in det_prediction
    ]
    # log results
    draw_bboxes(Image.fromarray(img), ocr_bboxes, name="ocr_bboxes(unmerged).png")

    ocr_bboxes = sorted(ocr_bboxes)
    ocr_bboxes = bbox_merge(ocr_bboxes)
    # log results
    draw_bboxes(Image.fromarray(img), ocr_bboxes, name="ocr_bboxes(merged).png")
    ocr_bboxes = split_conflict(ocr_bboxes, latex_bboxes)
    ocr_bboxes = list(filter(lambda x: x.label == "text", ocr_bboxes))

    sliced_imgs: List[np.ndarray] = slice_from_image(img, ocr_bboxes)
    start_time = time.time()
    rec_predictions, _ = rec_model(sliced_imgs)
    end_time = time.time()
    print(f"ocr_rec_model time: {end_time - start_time:.2f}s")

    assert len(rec_predictions) == len(ocr_bboxes)
    for content, bbox in zip(rec_predictions, ocr_bboxes):
        bbox.content = content[0]
    
    latex_imgs =[]
    for bbox in latex_bboxes:
        latex_imgs.append(img[bbox.p.y:bbox.p.y + bbox.h, bbox.p.x:bbox.p.x + bbox.w])
    start_time = time.time()
    latex_rec_res = latex_rec_predict(*latex_rec_models, latex_imgs, accelerator, num_beams, max_tokens=800)
    end_time = time.time()
    print(f"latex_rec_model time: {end_time - start_time:.2f}s")

    for bbox, content in zip(latex_bboxes, latex_rec_res):
        bbox.content = to_katex(content)
        if bbox.label == "embedding":
            bbox.content = " $" + bbox.content + "$ "
        elif bbox.label == "isolated":
            bbox.content = '\n\n' + r"$$" + bbox.content + r"$$" + '\n\n'


    bboxes = sorted(ocr_bboxes + latex_bboxes)
    if bboxes == []:
        return ""

    md = ""
    prev = Bbox(bboxes[0].p.x, bboxes[0].p.y, -1, -1, label="guard")
    for curr in bboxes:
        # Add the formula number back to the isolated formula
        if (
            prev.label == "isolated"
            and curr.label == "text"
            and prev.same_row(curr)
        ):
            curr.content = curr.content.strip()
            if curr.content.startswith('(') and curr.content.endswith(')'):
                curr.content = curr.content[1:-1]

            if re.search(r'\\tag\{.*\}$', md[:-4]) is not None:
                # in case of multiple tag
                md = md[:-5] + f', {curr.content}' + '}' + md[-4:]
            else:
                md = md[:-4] + f'\\tag{{{curr.content}}}' + md[-4:]
            continue

        if not prev.same_row(curr):
            md += " "

        if curr.label == "embedding":
            # remove the bold effect from inline formulas
            curr.content = change_all(curr.content, r'\bm', r' ', r'{', r'}', r'', r' ')
            curr.content = change_all(curr.content, r'\boldsymbol', r' ', r'{', r'}', r'', r' ')
            curr.content = change_all(curr.content, r'\textit', r' ', r'{', r'}', r'', r' ')
            curr.content = change_all(curr.content, r'\textbf', r' ', r'{', r'}', r'', r' ')
            curr.content = change_all(curr.content, r'\textbf', r' ', r'{', r'}', r'', r' ')
            curr.content = change_all(curr.content, r'\mathbf', r' ', r'{', r'}', r'', r' ')

            # change split environment into aligned
            curr.content = curr.content.replace(r'\begin{split}', r'\begin{aligned}')
            curr.content = curr.content.replace(r'\end{split}', r'\end{aligned}')

            # remove extra spaces (keeping only one)
            curr.content = re.sub(r' +', ' ', curr.content)
            assert curr.content.startswith(' $') and curr.content.endswith('$ ')
            curr.content = ' $' + curr.content[2:-2].strip() + '$ '
        md += curr.content
        prev = curr
    return md.strip()
