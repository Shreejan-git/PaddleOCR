# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import importlib

__dir__ = os.path.dirname(__file__)

from typing import List, Any, Union

# import paddle
from ocr.get_utils import parse_lang, parse_args, get_model_config, check_img
from ocr.paddle_ocr_settings import SUPPORT_OCR_MODEL_VERSION, SUPPORT_DET_MODEL, SUPPORT_REC_MODEL
from ocr.paddle_structure import PPStructure

# from paddleocr import check_img, parse_args

sys.path.append(os.path.join(__dir__, ''))

import cv2
import logging
import numpy as np
from pathlib import Path


# import base64
# from io import BytesIO
# from PIL import Image


def _import_file(module_name, file_path, make_importable=False):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if make_importable:
        sys.modules[module_name] = module
    return module


tools = _import_file(
    'tools', os.path.join(__dir__, '../tools/__init__.py'), make_importable=True)

ppocr = importlib.import_module('ppocr', 'paddleocr')
ppstructure = importlib.import_module('ppstructure', 'paddleocr')
from ppocr.utils.logging import get_logger
from tools.infer import predict_system
from ppocr.utils.utility import check_and_read, get_image_file_list, alpha_to_color, binarize_img
from ppocr.utils.network import maybe_download, download_with_progressbar, is_link, confirm_model_dir_url
from tools.infer.utility import draw_ocr, str2bool, check_gpu
from ppstructure.utility import init_args, draw_structure_result
from ppstructure.predict_system import StructureSystem, save_structure_res, to_excel

logger = get_logger()
__all__ = [
    'PaddleOCR', 'PPStructure', 'draw_ocr', 'draw_structure_result',
    'save_structure_res', 'download_with_progressbar', 'to_excel'
]

BASE_DIR = os.path.expanduser("~/.paddleocr/")


class PaddleOCR(predict_system.TextSystem):
    def __init__(self, lang='en', det='DB', rec='', type_='ocr', use_gpu=False, **kwargs):
        """
        paddleocr package
        args:x
            **kwargs: other params show in paddleocr --help
        """
        params = parse_args(mMain=False)  # parse_args ko import crosscheck
        params.__dict__.update(**kwargs)
        assert params.ocr_version in SUPPORT_OCR_MODEL_VERSION, "ocr_version must in {}, but get {}".format(
            SUPPORT_OCR_MODEL_VERSION, params.ocr_version)
        params.use_gpu = check_gpu(use_gpu)

        if not params.show_log:
            logger.setLevel(logging.INFO)
        self.use_angle_cls = params.use_angle_cls
        lang, det_lang = parse_lang(lang)

        # init model dir
        det_model_config = get_model_config('OCR', params.ocr_version, 'det',
                                            det_lang)
        params.det_model_dir, det_url = confirm_model_dir_url(
            params.det_model_dir,
            os.path.join(BASE_DIR, 'whl', 'det', det_lang),
            det_model_config['url'])

        # init model recognition
        rec_model_config = get_model_config('OCR', params.ocr_version, 'rec',
                                            lang)
        params.rec_model_dir, rec_url = confirm_model_dir_url(
            params.rec_model_dir,
            os.path.join(BASE_DIR, 'whl', 'rec', lang), rec_model_config['url'])

        """
        cls_model_config = get_model_config('OCR', params.ocr_version, 'cls',
                                            'en')
        params.cls_model_dir, cls_url = confirm_model_dir_url(
            params.cls_model_dir,
            os.path.join(BASE_DIR, 'whl', 'cls'), cls_model_config['url'])
        """

        if params.ocr_version in ['PP-OCRv3', 'PP-OCRv4']:
            params.rec_image_shape = "3, 48, 320"
        else:
            params.rec_image_shape = "3, 32, 320"

        # download model if using paddle infer
        if not params.use_onnx:
            maybe_download(params.det_model_dir, det_url)
            maybe_download(params.rec_model_dir, rec_url)
            # maybe_download(params.cls_model_dir, cls_url)

        if params.det_algorithm not in SUPPORT_DET_MODEL:
            logger.error('det_algorithm must in {}'.format(SUPPORT_DET_MODEL))
            sys.exit(0)

        if params.rec_algorithm not in SUPPORT_REC_MODEL:
            logger.error('rec_algorithm must in {}'.format(SUPPORT_REC_MODEL))
            sys.exit(0)

        if params.rec_char_dict_path is None:
            params.rec_char_dict_path = str(
                Path(__file__).parent / rec_model_config['dict_path'])

        logger.debug(params)
        # init det_model and rec_model
        super().__init__(params)
        self.page_num = params.page_num

    def ocr(self,
            img,
            det=True,
            rec=False,
            cls=False,
            bin=False,
            inv=False,
            alpha_color=(255, 255, 255)):
        """
        OCR with PaddleOCR
        args：
            img: img for OCR, support ndarray, img_path and list or ndarray
            det: use text detection or not. If False, only rec will be exec. Default is True
            rec: use text recognition or not. If False, only det will be exec. Default is True
            cls: use angle classifier or not. Default is True. If True, the text with rotation of 180 degrees can be recognized. If no text is rotated by 180 degrees, use cls=False to get better performance. Text with rotation of 90 or 270 degrees can be recognized even if cls=False.
            bin: binarize image to black and white. Default is False.
            inv: invert image colors. Default is False.
            alpha_color: set RGB color Tuple for transparent parts replacement. Default is pure white.
        """
        assert isinstance(img, (np.ndarray, list, str, bytes))
        if isinstance(img, list) and det == True:
            logger.error('When input a list of images, det must be false')
            exit(0)
        if cls == True and self.use_angle_cls == False:
            logger.warning(
                'Since the angle classifier is not initialized, it will not be used during the forward process'
            )

        # img = check_img(img)
        # for infer pdf file
        if isinstance(img, list):
            if self.page_num > len(img) or self.page_num == 0:
                self.page_num = len(img)
            imgs = img[:self.page_num]
        else:
            imgs = [img]

        def preprocess_image(_image):
            _image = alpha_to_color(_image, alpha_color)
            if inv:
                _image = cv2.bitwise_not(_image)
            if bin:
                _image = binarize_img(_image)
            return _image

        if det and rec:
            ocr_res = []
            for idx, img in enumerate(imgs):
                # img = preprocess_image(img)
                dt_boxes, rec_res, _ = self.__call__(img, cls)
                if not dt_boxes and not rec_res:
                    ocr_res.append(None)
                    continue
                tmp_res = [[box.tolist(), res]
                           for box, res in zip(dt_boxes, rec_res)]
                ocr_res.append(tmp_res)
            return ocr_res
        elif det and not rec:
            ocr_res = []
            for idx, img in enumerate(imgs):
                img = preprocess_image(img)
                dt_boxes, elapse = self.text_detector(img)
                if not dt_boxes:
                    ocr_res.append(None)
                    continue
                tmp_res = [box.tolist() for box in dt_boxes]
                ocr_res.append(tmp_res)
            return ocr_res
        else:
            ocr_res = []
            cls_res = []git
            for idx, img in enumerate(imgs):
                if not isinstance(img, list):
                    img = preprocess_image(img)
                    img = [img]
                if self.use_angle_cls and cls:
                    img, cls_res_tmp, elapse = self.text_classifier(img)
                    if not rec:
                        cls_res.append(cls_res_tmp)
                rec_res, elapse = self.text_recognizer(img)
                ocr_res.append(rec_res)
            if not rec:
                return cls_res
            return ocr_res


def preprocess_input(input_path: str) -> Union[List[Any], bool]:
    """
    ar
    """
    image_dir = input_path

    if is_link(image_dir):
        download_with_progressbar(image_dir, 'tmp.jpg')
        image_file_list = ['tmp.jpg']
    else:
        image_file_list = get_image_file_list(image_dir)
    if len(image_file_list) == 0:
        logger.error('no images find in {}'.format(image_dir))
        return image_file_list

    return image_file_list


def paddle_det_and_rec(input_path, det=True, rec=False):
    args = parse_args(mMain=True)

    image_file_list = preprocess_input(input_path)
    engine = PaddleOCR(**args.__dict__)

    if len(image_file_list) > 0:
        for img_path in image_file_list:

            img = check_img(img_path)

            result = engine.ocr(img,
                                det=det,
                                rec=rec)
            print(result)
            # img_name = os.path.basename(img_path).split('.')[0]


if __name__ == '__main__':
    input_path = '/home/vertexml/Documents/test_images/tabular_image_data.jpg'
    paddle_det_and_rec(input_path=input_path)
