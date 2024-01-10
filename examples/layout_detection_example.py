import os
import random
from typing import List, Tuple, Dict, Union

import cv2
import numpy as np

from ocr import parse_args, PPStructure
from ppocr.utils.network import is_link, download_with_progressbar
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.utils.logging import get_logger

logger = get_logger()


def paddle_layout_table_extraction(file_path: str):
    args = parse_args(mMain=True)

    if is_link(file_path):
        download_with_progressbar(file_path, 'tmp.jpg')
        file_list = ['tmp.jpg']
    else:
        file_list: List[str] = get_image_file_list(file_path)  # list containing path of each pdf files
    if len(file_list) == 0:
        logger.error('no images find in {}'.format(file_path))
        return
    engine = PPStructure(**args.__dict__)

    for file_path in file_list:  # looping in each file
        img_name = os.path.basename(file_path).split('.')[0]  # file name
        logger.info('{}{}{}'.format('*' * 10, file_path, '*' * 10))

        total_page_count, img, flag_gif, flag_pdf = check_and_read(file_path)

        if not flag_gif and not flag_pdf:
            img = cv2.imread(file_path)

        if not flag_pdf:
            if img is None:
                logger.error("error in loading image:{}".format(file_path))
                continue
            img_paths: List[List[Union[str, np.ndarray]]] = [[file_path, img]]
        else:
            img_paths: List[List[Union[str, np.ndarray]]] = []  # yo aile chaiye ko chai xaina.
            for index, pdf_img in enumerate(img, start=1):
                os.makedirs(
                    os.path.join(args.output, img_name), exist_ok=True)
                pdf_img_path = os.path.join(
                    args.output, img_name,
                    img_name + '_' + str(index) + '.jpg')
                cv2.imwrite(pdf_img_path, pdf_img)
                img_paths.append([pdf_img_path, pdf_img])

        for index, (new_img_path, img) in enumerate(img_paths, start=1):  # looping in each page
            logger.info('processing {}/{} page:'.format(index + 1,
                                                        len(img_paths)))
            img_name: str = os.path.basename(new_img_path).split('.')[0]
            page_results: List[Dict] = engine(img, img_idx=index)  # List containing information of a given page in a
            # dictionary. Each dictionary contains information of each detected layout.

            all_recognized_text = []  # list containing all recognized texts.
            all_confidence_scores = []  # list of all confidence scores of all_recognized_text

            if page_results:
                for layout in page_results:  # looping on each detected layout
                    red = random.randint(0, 256)
                    green = random.randint(0, 256)
                    blue = random.randint(0, 256)

                    l, t, r, b = layout['bbox']  # ltrb format of layouts detected layout's bounding box
                    cropped_img: np.ndarray = layout['img']  # each detected layout's cropped part
                    cls_type: str = layout['type']
                    recog_info: Dict = layout['res']
                    # if recog_info:
                    #     word_bbox: List[List] = recog_info[
                    #         'boxes']  # sl = sub-left. bboxes of each detection word with in
                    #     # the layout.
                    #     rec_res: List[Tuple] = recog_info['rec_res']  # list with recognized words and confidence
                    #     for text_conf in rec_res:
                    #         recognized_words: str = text_conf[0]  # recognized words
                    #         confidence: float = text_conf[1]  # confidence score of each recognized word
                    #         all_recognized_text.append(recognized_words)
                    #         all_confidence_scores.append(confidence)
                    #     if cls_type == "table":
                    #         html_tags: str = recog_info['html']  # html tags of the tabular data
                    # image_idx = layout['img_idx']

                    cv2.rectangle(img, (l, t), (r, b), [blue, green, red], 2)
                    cv2.putText(img, cls_type, (l, t + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, [blue, green, red], 2,
                                cv2.LINE_AA)

                cv2.namedWindow('Layout Prediction', cv2.WINDOW_NORMAL)
                cv2.imshow('Layout Prediction', img)
                cv2.waitKey(0)


if __name__ == "__main__":
    # file_path = "/home/vertexaiml/Downloads/ocr_test_image/test_invoice.png" # whole lai figure vanirako xa.
    # file_path = "/home/vertexaiml/Downloads/Vertex_It/Poc_Sample/US_Bank/01.2021-01-29 Statement - USB Y _ Y INC...4004.pdf"
    # file_path = "/home/vertexaiml/Downloads/Vertex_It/Poc_Sample/Wellsfargo/Wellsfargo.pdf"  # table detection
    # ramro xa
    # file_path = "/home/vertexaiml/Downloads/Vertex_It/Poc_Sample/Wellsfargo/Wells Fargo -Apr 2022.pdf"
    file_path = "/home/vertexaiml/Downloads/Vertex_It/Poc_Sample/Bank_Of_America/Bank of America.pdf"
    paddle_layout_table_extraction(file_path=file_path)
