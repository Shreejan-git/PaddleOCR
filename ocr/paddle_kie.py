import os
from pathlib import Path
import logging
import random

import cv2

from ocr.config import parse_args, SUPPORT_STRUCTURE_MODEL_VERSION, parse_lang, get_model_config
from ocr.utils import check_img
from ppocr.utils.network import confirm_model_dir_url, maybe_download, is_link, download_with_progressbar
from ppocr.utils.utility import check_and_read, get_image_file_list
from ppstructure.predict_system import StructureSystem, save_structure_res

from tools.infer.utility import check_gpu
from ppocr.utils.logging import get_logger

logger = get_logger()
BASE_DIR = os.path.expanduser("~/.paddleocr/")


class PPStructure(StructureSystem):
    def __init__(self, lang='en', use_gpu=False, layout=True, table=False, **kwargs):
        params = parse_args(mMain=False)
        params.__dict__.update(**kwargs)
        assert params.structure_version in SUPPORT_STRUCTURE_MODEL_VERSION, "structure_version must in {}, but get {}".format(
            SUPPORT_STRUCTURE_MODEL_VERSION, params.structure_version)
        params.use_gpu = check_gpu(use_gpu)
        params.mode = "kie"

        if not params.show_log:
            logger.setLevel(logging.INFO)
        lang, det_lang = parse_lang(lang)
        if lang == 'ch':
            table_lang = 'ch'
        else:
            table_lang = 'en'
        if params.structure_version == 'PP-Structure':
            params.merge_no_span_structure = False

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

        table_model_config = get_model_config(
            'STRUCTURE', params.structure_version, 'table', table_lang)
        params.table_model_dir, table_url = confirm_model_dir_url(
            params.table_model_dir,
            os.path.join(BASE_DIR, 'whl', 'table'), table_model_config['url'])

        layout_model_config = get_model_config(
            'STRUCTURE', params.structure_version, 'layout', lang)
        params.layout_model_dir, layout_url = confirm_model_dir_url(
            params.layout_model_dir,
            os.path.join(BASE_DIR, 'whl', 'layout'), layout_model_config['url'])
        # download model
        maybe_download(params.det_model_dir, det_url)
        maybe_download(params.rec_model_dir, rec_url)
        maybe_download(params.table_model_dir, table_url)
        maybe_download(params.layout_model_dir, layout_url)

        if params.rec_char_dict_path is None:
            params.rec_char_dict_path = '/home/vertexaiml/Documents/PaddleOCR/ppocr/utils/en_dict.txt'

            # params.rec_char_dict_path = str(
            #     Path(__file__).parent / rec_model_config['dict_path'])

        if params.table_char_dict_path is None:
            params.table_char_dict_path = "/home/vertexaiml/Documents/PaddleOCR/ppocr/utils/dict/table_structure_dict.txt"
            # params.table_char_dict_path = str(
            #     Path(__file__).parent / table_model_config['dict_path'])

        if params.layout_dict_path is None:
            params.layout_dict_path = ("/home/vertexaiml/Documents/PaddleOCR/ppocr/utils/dict/layout_dict"
                                       "/layout_publaynet_dict.txt")

            # params.layout_dict_path = str(
            #     Path(__file__).parent / layout_model_config['dict_path'])

        logger.debug(params)
        super().__init__(params, layout=layout, table=table, ocr=True)

    def __call__(self, img, return_ocr_result_in_table=True, img_idx=0):
        img = check_img(img)
        res, _ = super().__call__(
            img, return_ocr_result_in_table, img_idx=img_idx)

        return res


def paddle_layout_table_extraction(file_path: str):
    args = parse_args(mMain=True)
    if is_link(file_path):
        download_with_progressbar(file_path, 'tmp.jpg')
        image_file_list = ['tmp.jpg']
    else:
        image_file_list = get_image_file_list(file_path)
    if len(image_file_list) == 0:
        logger.error('no images find in {}'.format(file_path))
        return
    engine = PPStructure(**args.__dict__)

    for img_path in image_file_list:
        img_name = os.path.basename(img_path).split('.')[0]
        logger.info('{}{}{}'.format('*' * 10, img_path, '*' * 10))

        img, flag_gif, flag_pdf = check_and_read(img_path)
        if not flag_gif and not flag_pdf:
            img = cv2.imread(img_path)

        if not flag_pdf:
            if img is None:
                logger.error("error in loading image:{}".format(img_path))
                continue
            img_paths = [[img_path, img]]
        else:
            img_paths = []
            for index, pdf_img in enumerate(img):
                os.makedirs(
                    os.path.join(args.output, img_name), exist_ok=True)
                pdf_img_path = os.path.join(
                    args.output, img_name,
                    img_name + '_' + str(index) + '.jpg')
                cv2.imwrite(pdf_img_path, pdf_img)
                img_paths.append([pdf_img_path, pdf_img])

        # all_res = []
        for index, (new_img_path, img) in enumerate(img_paths):
            logger.info('processing {}/{} page:'.format(index + 1,
                                                        len(img_paths)))
            new_img_name = os.path.basename(new_img_path).split('.')[0]
            results = engine(img, img_idx=index)

            # save_structure_res(results, args.output, img_name, index)

            for layout in results:
                # print(layout)
                red = random.randint(0, 256)
                green = random.randint(0, 256)
                blue = random.randint(0, 256)
                l, t, r, b = layout['bbox']  # ltrb format
                cropped_img = layout['img']
                cls_type = layout['type']
                recog_text = layout['res']

                cv2.rectangle(img, (l, t), (r, b), [blue, green, red], 2)
                cv2.putText(img, cls_type, (l, t), cv2.FONT_HERSHEY_SIMPLEX, 1, [blue, green, red], 2, cv2.LINE_AA)

            cv2.namedWindow('Layout Prediction', cv2.WINDOW_NORMAL)
            cv2.imshow('Layout Prediction', img)
            cv2.waitKey(0)


if __name__ == "__main__":
    file_path = "/home/vertexaiml/Downloads/ocr_test_image/4 page.jpg"
    file_path = "/home/vertexaiml/Downloads/ocr_test_image/test_invoice.png"
    file_path = "/home/vertexaiml/Downloads/ocr_test_image/layout_testing_input.png"
    paddle_layout_table_extraction(file_path=file_path)
