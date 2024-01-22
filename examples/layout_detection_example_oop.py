import os
import random
from typing import List, Tuple, Dict, Union

import cv2
import numpy as np

from ocr import parse_args, PPStructure
from ppocr.utils.network import is_link, download_with_progressbar
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.utils.logging import get_logger
from multiprocessing import Pool

logger = get_logger()


class VertexOCR:

    def __init__(self):
        self.args = parse_args(mMain=True)
        self.engine = PPStructure(**self.args.__dict__)

    def extract_layout(self, input_file: str, visualize=False):

        if is_link(input_file):
            download_with_progressbar(input_file, 'tmp.jpg')
            file_list = ['tmp.jpg']
        else:
            file_list: List[str] = get_image_file_list(input_file)  # list containing path of each pdf files
        if len(file_list) == 0:
            logger.error('no images find in {}'.format(input_file))
            return

        for input_file in file_list:  # looping in each file if multiple pdf files are submitted.
            img_name = os.path.basename(input_file).split('.')[0]  # file name
            logger.info('{}{}{}'.format('*' * 10, input_file, '*' * 10))

            total_page_count, imgs, flag_gif, flag_pdf = check_and_read(input_file)  # convert pdf to image
            # imgs is a list. It has pages' image form in an original order.

            if not flag_gif and not flag_pdf:
                imgs = cv2.imread(input_file)

            if not flag_pdf:
                if imgs is None:
                    logger.error("error in loading image:{}".format(input_file))
                    continue
                # img_paths: List[List[Union[str, np.ndarray]]] = [[file_path, imgs]]
            else:
                with Pool(processes=5) as pool:
                    results = pool.apply_async(self.engine, imgs)
                    print(results)

                    # print(results.__next__)

                    # for i, j in results:
                    #     print(f"{i} completed in {j} time")

                # IMPORTANT !!!
                # PDF ho vane matrai multiprocessing lagaunu parxa or input nai multiple images ho vane matrai.
                # single image matrai input ayeko xa vaye 1 core use garnu parne xa.
                '''
                img_paths: List[List[Union[str, np.ndarray]]] = []  # yo aile chaiye ko chai xaina.
                for index, pdf_img in enumerate(imgs, start=1):
                    os.makedirs(
                        os.path.join(self.args.output, img_name), exist_ok=True)
                    pdf_img_path = os.path.join(
                        self.args.output, img_name,
                        img_name + '_' + str(index) + '.jpg')
                    # cv2.imwrite(pdf_img_path, pdf_img)
                    img_paths.append([pdf_img_path, pdf_img])
                '''
            # layout_result = []
            # for index, (new_img_path, img) in enumerate(img_paths, start=0):  # looping in each page
            #     logger.info('processing {}/{} page:'.format(index + 1,
            #                                                 len(img_paths)))
            #     # img_name: str = os.path.basename(new_img_path).split('.')[0]
            #     page_results: List[Dict] = self.engine(img, img_idx=index)
            #
            #     all_recognized_text = []  # list containing all recognized texts.
            #     all_confidence_scores = []  # list of all confidence scores of all_recognized_text
            #
            #     if page_results:
            #         for layout in page_results:  # looping on each detected layout
            #             red = random.randint(0, 256)
            #             green = random.randint(0, 256)
            #             blue = random.randint(0, 256)
            #
            #             cls_type: str = layout['type']  # layout type
            #             l, t, r, b = layout['bbox']  # ltrb formatted bbox of detected layout
            #             cropped_img: np.ndarray = layout['img']  # detected layout's cropped part
            #             recog_info: Union[list[dict], dict] = layout[
            #                 'res']  # list of dicts of detected texts with its confidence
            #             image_index = layout['img_idx']
            #
            #             if recog_info:
            #                 if cls_type == "table":
            #                     cell_bboxes: list = recog_info['cell_bbox']  # bbox of each cell within the table
            #                     bboxes: list = recog_info['boxes']  # bbox of each text detected within the table
            #                     rec_res: list[tuple] = recog_info[
            #                         'rec_res']  # tuple of recognized texts with confidence
            #                     for text, confidence in rec_res:
            #                         recognized_words: str = text
            #                         confidence: float = confidence
            #                         all_recognized_text.append(recognized_words)
            #                         all_confidence_scores.append(confidence)
            #
            #                     html_tags: str = recog_info['html']  # html tags of the tabular data
            #
            #                 else:
            #                     for text_conf in recog_info:
            #                         recognized_words: str = text_conf['text']  # recognized texts
            #                         confidence: float = text_conf[
            #                             'confidence']  # confidence score of each recognized text
            #                         text_bbox: list = text_conf['text_region']  # bboxes of each detected text within
            #                         # the detected layout.
            #                         all_recognized_text.append(recognized_words)
            #                         all_confidence_scores.append(confidence)
            #
            #                 if visualize:
            #                     cv2.rectangle(img, (l, t), (r, b), [blue, green, red], 2)
            #                     cv2.putText(img, cls_type, (l, t + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, [blue, green, red],
            #                                 2,
            #                                 cv2.LINE_AA)
            #         if visualize:
            #             cv2.namedWindow('Layout Prediction', cv2.WINDOW_NORMAL)
            #             cv2.imshow('Layout Prediction', img)
            #             cv2.waitKey(0)


if __name__ == "__main__":
    file_path = "/home/vertexaiml/Downloads/Vertex_It/Poc_Sample/US_Bank/01.2021-01-29 Statement - USB Y _ Y INC...4004.pdf"
    # file_path = "/home/vertexaiml/Downloads/Vertex_It/Poc_Sample/Wellsfargo/wellsfargo_pdf"
    # paddle_layout_table_extraction(file_path=file_path, visualize=True)
    vertex_ocr = VertexOCR()
    vertex_ocr.extract_layout(input_file=file_path)
