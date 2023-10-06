import os
import sys
from pathlib import Path
import logging
import cv2
import numpy as np
import importlib
from ocr.utils import _import_file

# __dir__ = os.path.dirname(__file__)
# tools = _import_file(
#     'tools', os.path.join(__dir__, 'tools/__init__.py'), make_importable=True)

# ppocr = importlib.import_module('ppocr', 'paddleocr')

from ppocr.utils.network import maybe_download, download_with_progressbar, is_link, confirm_model_dir_url
from ppocr.utils.utility import alpha_to_color, binarize_img, get_image_file_list
from tools.infer import predict_system
from ocr.config import parse_args, SUPPORT_OCR_MODEL_VERSION, parse_lang, get_model_config, \
    SUPPORT_DET_MODEL, SUPPORT_REC_MODEL
from ocr.utils import check_img, _import_file
from tools.infer.utility import draw_ocr, str2bool, check_gpu
from ocr.config import BASE_DIR

from ppocr.utils.logging import get_logger

logger = get_logger()

__all__ = [
    'PaddleOCR', 'draw_ocr', 'download_with_progressbar'
]


# print(f"BASE_DIR VALUE: {BASE_DIR}")


class PaddleOCR(predict_system.TextSystem):
    def __init__(self, lang='en', use_gpu=False, **kwargs):
        """
        paddleocr package
        args:
            **kwargs: other params show in paddleocr --help
        """
        params = parse_args(mMain=False)
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
        rec_model_config = get_model_config('OCR', params.ocr_version, 'rec',
                                            lang)
        params.rec_model_dir, rec_url = confirm_model_dir_url(
            params.rec_model_dir,
            os.path.join(BASE_DIR, 'whl', 'rec', lang), rec_model_config['url'])
        cls_model_config = get_model_config('OCR', params.ocr_version, 'cls',
                                            'ch')
        params.cls_model_dir, cls_url = confirm_model_dir_url(
            params.cls_model_dir,
            os.path.join(BASE_DIR, 'whl', 'cls'), cls_model_config['url'])
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
            params.rec_char_dict_path = '/home/vertexaiml/Documents/PaddleOCR/ppocr/utils/en_dict.txt'
            # params.rec_char_dict_path = str(
            #     Path(__file__).parent / rec_model_config['dict_path'])

        logger.debug(params)
        # init det_model and rec_model
        super().__init__(params)
        self.page_num = params.page_num

    def ocr(self,
            img,
            det=True,
            rec=True,
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

        img = check_img(img)
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
                img = preprocess_image(img)
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
                # img = preprocess_image(img)
                dt_boxes, elapse = self.text_detector(img)
                if len(dt_boxes) == 0:
                    ocr_res.append(None)
                    continue
                tmp_res = [box.tolist() for box in dt_boxes]
                ocr_res.append(tmp_res)
            return ocr_res
        else:
            ocr_res = []
            cls_res = []
            for idx, img in enumerate(imgs):
                if not isinstance(img, list):
                    # img = preprocess_image(img)
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


def paddle_det_and_rec(input_file=None, det=True, rec=False):
    args = parse_args(mMain=True)

    if is_link(input_file):
        download_with_progressbar(input_file, 'tmp.jpg')
        image_file_list = ['tmp.jpg']
    else:
        image_file_list = get_image_file_list(input_file)
    if len(image_file_list) == 0:
        logger.error('no images find in {}'.format(input_file))
        return

    engine = PaddleOCR(**args.__dict__)

    for img_path in image_file_list:
        img_name = os.path.basename(img_path).split('.')[0]
        logger.info('{}{}{}'.format('*' * 10, img_path, '*' * 10))
        result = engine.ocr(img_path,
                            det=det,
                            rec=rec,
                            cls=args.use_angle_cls,
                            bin=args.binarize,
                            inv=args.invert,
                            alpha_color=args.alphacolor)
        if det and rec:
            for rec_result in result[0]:
                print(rec_result)

        elif det and not rec:
            if result is not None:
                # for idx in range(len(result)):
                #     res = result[idx]
                #     for line in res:
                #         logger.info(line)

                visualize_bboxes(img_path, result)
            else:
                print("Could not detect anything")


def visualize_bboxes(img_path, result):
    img = cv2.imread(img_path)
    result = result[0]
    for bbox in result:
        # print(bbox)
        l, t = bbox[0]
        r, b = bbox[2]

        img = cv2.rectangle(img, (int(l), int(t)), (int(r), int(b)), [255, 0, 0], 2)

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)
    cv2.waitKey()


if __name__ == "__main__":

    image_path = "/home/vertexaiml/Downloads/ocr_test_image/nepal.png"
    # image_path = "/home/vertexaiml/Downloads/4 page.jpg"
    # image_path = "/home/vertexaiml/Downloads/ocr_test_image/blank_white.jpg"
    paddle_det_and_rec(input_file=image_path, det=True, rec=True)