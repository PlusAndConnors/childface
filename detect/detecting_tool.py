"""
Run a rest API exposing the retinaface object detection model
"""
import argparse
import csv
import io
import json
import os
import os.path as osp
from typing import Tuple, Dict, List

import cv2
import numpy as np
import torch  # model typehint: nn.Module
from PIL import Image  # typehint: Image.Image
from numpy.typing import NDArray
from skimage import transform as trans
from tqdm import tqdm

from detect.models.retinaface import RetinaFace
from .tool import cfg_re50
from .tool.prior_box import PriorBox
from .utils.box_utils import decode, decode_landm
from .utils.nms.py_cpu_nms import py_cpu_nms


# TODO : use assert -> if error post message using try
# TODO : tpyint show args type

class UseOurDetect:
    def __init__(self, args=None):
        if args is None:
            parser = argparse.ArgumentParser(description='Retinaface')
            parser.add_argument('-m', '--trained_detect_model',
                                default=osp.join('detect', 'models', 'Resnet50_Final.pth'), type=str,
                                help='Trained state_dict file path to open')
            parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
            parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
            parser.add_argument('--top_k', default=5000, type=int, help='top_k')
            parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
            parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
            parser.add_argument('--vis_threshold', default=0.6, type=float, help='visualization_threshold')
            self.args = parser.parse_args()
        else:
            self.args = args

        if get_gpu_memory_map()[torch.cuda.current_device()] > 40000:
            device = torch.device("cuda:1")
            torch.cuda.set_device(device)
        device = torch.cuda.current_device()
        net = RetinaFace(cfg=cfg_re50, phase='test')
        net = load_model(net, args.trained_detect_model, args.cpu)
        net = net.to(device)
        net.eval()
        self.net = net
        self.device = device

    def detect_with_retinaface(self, img_path: NDArray) -> Tuple[Dict, Dict, List]:
        args, results, crop_imgs, result, bbox = self.args, dict(), dict(), list(), list()

        img_raw, dets = self.process_retinaface(img_path)
        for ind, box in enumerate(dets):
            if box[4] < args.vis_threshold:
                continue
            # for conf
            text = "{:.4f}".format(box[4])
            box = list(map(int, box))

            # for crop img
            for i, v in enumerate(box[0:4]):
                if v < 0:
                    box[i] = 0
            # img_raw has a type of PIL or cv2 depending on det or rec
            if 'PIL' in str(type(img_raw)):
                one_crop_img = img_raw.crop((box[0], box[1], box[2], box[3]))
            else:
                one_crop_img = img_raw[box[1]: box[3], box[0]: box[2]]
            crop_imgs[ind] = one_crop_img
            # landms
            for i in range(5):
                box[2 * i + 5] = box[2 * i + 5] - box[0]  # 5 7 9 11 13
                box[2 * i + 6] = box[2 * i + 6] - box[1]  # 6 8 10 12 14
            result.append(result_sort(ind, [box[5:15], text])), bbox.append(box[0:4])
        results['landmk_info'] = result
        return results, crop_imgs, bbox

    def newcls_feature_init_(self):
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        src[:, 0] += 8.0
        self.src = src
        self.fine_error = [UnboundLocalError, AssertionError]

    def new_img2bbox_annotation(self, image_file: List):
        '''
        Args:
            image_file: img_file (post img or json)
            "image": [base64 encoding 된 이미지1, base64 encoding 된 이미지2 .. ]
        Returns:
            bbox = [x,y,x,y,conf]
            crop_img = NDarray
            result = '0.jpg 70 106 149 118 102 144 66 188 128 199 0.9997' | landmark
        '''
        self.newcls_feature_init_()
        imgs, bboxs, results = list(), list(), list()
        for base64_img in image_file:
            cv_img = base642cv(base64_img)
            imgs.append(cv_img)
        print(f'{len(imgs)} images')
        for i, raw_img in enumerate(tqdm(imgs)):
            try:
                # bbox info + image
                bbox, result = self.extract_one_landmark(raw_img)
                bboxs.append(bbox), results.append(result)
            except self.fine_error:  # detect model can't find landmark
                pass
        return bboxs, results, imgs

    def extract_one_landmark(self, img: NDArray):
        img_raw, dets = self.process_retinaface(img)
        results, conf, result = dict(), 0, 0
        if not len(dets):
            print(f'.. retinaface hard to detect this face : {img}')
            bbox, crop_img = 0, np.array(0)
        else:
            high_score_index = np.argmax(dets[:, 4])
            b = dets[high_score_index]
            # conf
            conf = max(b[4], conf)
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            # crop img
            for i, v in enumerate(b[0:4]):
                if v < 0:
                    b[i] = 0
            result = result_sort(0, [b[5:15], text])
            bbox = b[0:4] + [conf]
            # crop_img = apply_affine(img_raw, b[5:15], self.src)
            results['landmk_info'] = result
        return bbox, result

    def process_retinaface(self, img_path: NDArray) -> Tuple[NDArray, NDArray]:
        torch.set_grad_enabled(False)
        args, cfg, resize = self.args, cfg_re50, 1
        # net and model
        img, scale, img_raw, im_height, im_width = collect_imgset_for_preprocess(img_path)
        img = img.to(self.device)
        scale = scale.to(self.device)
        loc, conf, landms = self.net(img)  # forward pass

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)

        boxes = decode(loc.data.squeeze(0), priors.data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), priors.data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        return img_raw, dets


def add_append(a, b):
    for i in b:
        a[i].append(i)


def collect_imgset_for_preprocess(img_raw: NDArray):
    if isinstance(img_raw, str):
        img_raw = cv2.imread(img_raw, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    img = np.float32(img_raw)

    im_height, im_width, _ = img.shape

    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    return img, scale, img_raw, im_height, im_width


def make_jsonset_for_preprocess(file: str) -> Dict[str, list]:
    if file.endswith('.json'):  # json_path
        with open(file, "r") as st_json:
            file = json.load(st_json)
    else:
        file = json.loads(file)  # json_file
    return file


import base64


def base642cv(img_file_base64):
    img_byte = base64.b64decode(img_file_base64)
    im_arr = np.frombuffer(img_byte, dtype=np.uint8)
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    return img


def show_result_img(img):
    img = cv2.imencode('.png', img)[1].tostring()
    f = io.BytesIO()
    f.write(img)
    f.seek(0)
    return f


def check_keys(model, pretrained_state_dict: Dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys

    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix: str) -> dict:
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path: str, load_to_cpu: bool):
    # print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def result_sort(ind: int, line: list) -> str:
    name = f'{ind}.jpg '
    for (index, value) in enumerate(line):
        if isinstance(value, list):
            for i in value:
                name = name + str(i)
                name = name + ' '
        else:
            name = name + str(value)
    return name


def get_gpu_memory_map() -> dict:
    import subprocess
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def apply_affine(rimg: NDArray, landmark5: list, src: NDArray) -> NDArray:
    landmark5 = np.array(landmark5).reshape((5, 2))
    tform = trans.SimilarityTransform()
    tform.estimate(landmark5, src)
    M = tform.params[0:2, :]
    img = cv2.warpAffine(rimg, M, (112, 112), borderValue=0.0)
    return img


def make_byte_image_2_cv2(image_file):
    image_bytes = image_file.read()
    image_cv = np.fromstring(image_bytes, np.uint8)
    img = cv2.imdecode(image_cv, cv2.IMREAD_COLOR)
    return img


def make_byte_image_2_PIL(image_file):
    image_bytes = image_file.read()
    img = Image.open(io.BytesIO(image_bytes))
    return img


# TODO
def draw_bbox_keypoint(re, bboxs, img) -> None:
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    for i, bbox in enumerate(bboxs):
        lmk = np.array([float(x) for x in re['landmk_info'][i].strip().split(' ')[1:-1]], dtype=np.float32)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        cv2.circle(img, (int(lmk[0]) + bbox[0], int(lmk[1]) + bbox[1]), 1, (0, 0, 255), 4)
        cv2.circle(img, (int(lmk[2]) + bbox[0], int(lmk[3]) + bbox[1]), 1, (0, 255, 255), 4)
        cv2.circle(img, (int(lmk[4]) + bbox[0], int(lmk[5]) + bbox[1]), 1, (255, 0, 255), 4)
        cv2.circle(img, (int(lmk[6]) + bbox[0], int(lmk[7]) + bbox[1]), 1, (0, 255, 0), 4)
        cv2.circle(img, (int(lmk[8]) + bbox[0], int(lmk[9]) + bbox[1]), 1, (255, 0, 0), 4)
        cv2.imwrite(f'./detect/test/{i}_{lmk[0]}_result.jpg', img)


# TODO
def write_csv(bbox, name, save_path) -> None:
    k = (0 if os.path.exists(f'{save_path}/input.csv') else 1)
    head = ['ID', 'FILE', 'FACE_X', 'FACE_Y', 'FACE_WIDTH', 'FACE_HEIGHT']
    with open(f'{save_path}/input.csv', 'a', newline='') as f:
        saveinfo = {'ID': 'H', 'FILE': f'images/{name}', 'FACE_X': bbox[0], 'FACE_Y': bbox[1],
                    'FACE_WIDTH': bbox[2] - bbox[0], 'FACE_HEIGHT': bbox[3] - bbox[1]}
        wr = csv.DictWriter(f, fieldnames=head)
        if k == 1:
            wr.writeheader()
        wr.writerow(saveinfo)
