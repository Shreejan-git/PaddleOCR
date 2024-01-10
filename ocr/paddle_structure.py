import os
import logging
from ocr.config import parse_args, SUPPORT_STRUCTURE_MODEL_VERSION, parse_lang, get_model_config
from ocr.utils import check_img
from ppocr.utils.network import confirm_model_dir_url, maybe_download
from ppstructure.predict_system import StructureSystem

from tools.infer.utility import check_gpu
from ppocr.utils.logging import get_logger

logger = get_logger()
BASE_DIR = os.path.expanduser("~/.paddleocr/")


class PPStructure(StructureSystem):
    def __init__(self, lang='en', use_gpu=True, layout=True, table=True, **kwargs):
        params = parse_args(mMain=False)
        params.__dict__.update(**kwargs)
        assert params.structure_version in SUPPORT_STRUCTURE_MODEL_VERSION, "structure_version must in {}, but get {}".format(
            SUPPORT_STRUCTURE_MODEL_VERSION, params.structure_version)
        params.use_gpu = check_gpu(use_gpu)
        params.mode = 'structure'

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

        current_file_absolute_dir_path = os.path.dirname(__file__)
        base = os.path.join(current_file_absolute_dir_path, '..', 'ppocr', 'utils')
        # print('value of base:', base)

        if params.rec_char_dict_path is None:
            file_path = os.path.join(base, 'en_dict.txt')
            if os.path.exists(file_path):
                params.rec_char_dict_path = file_path
            else:
                logger.info(f'INFO [Could not find rec_char_dict_path in {__file__} file]')

            # params.rec_char_dict_path = str(
            #     Path(__file__).parent / rec_model_config['dict_path'])

        if params.table_char_dict_path is None:
            # params.table_char_dict_path = "/home/vertexaiml/Documents/PaddleOCR/ppocr/utils/dict
            # /table_structure_dict.txt" params.table_char_dict_path = str( Path(__file__).parent /
            # table_model_config['dict_path'])
            file_path = os.path.join(base, 'dict', 'table_structure_dict.txt')
            if os.path.exists(file_path):
                params.table_char_dict_path = file_path
            else:
                logger.info(f'INFO [Could not find table_char_dict_path in {__file__} file]')

        if params.layout_dict_path is None:

            file_path = os.path.join(base, 'dict', 'layout_dict', 'layout_publaynet_dict.txt')
            if os.path.exists(file_path):
                params.layout_dict_path = file_path
            else:
                logger.info(f'INFO [Could not find table_char_dict_path in {__file__} file]')

            # params.layout_dict_path = ("/home/vertexaiml/Documents/PaddleOCR/ppocr/utils/dict/layout_dict"
            #                            "/layout_publaynet_dict.txt")

            # params.layout_dict_path = str(
            #     Path(__file__).parent / layout_model_config['dict_path'])

        # logger.debug(params)
        # print(params)
        super().__init__(params, layout=layout, table=table, ocr=True)

    def __call__(self, img, return_ocr_result_in_table=True, img_idx=0):
        img = check_img(img)  # blackNwhite image ho vane BGR ma lagxa navaye same image return hunxa
        res, _ = super().__call__(
            img, return_ocr_result_in_table, img_idx=img_idx)

        return res


if __name__ == "__main__":
    print(BASE_DIR)
    print(os.listdir(BASE_DIR))
