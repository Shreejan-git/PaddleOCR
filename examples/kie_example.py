from typing import Optional, List

import cv2
import numpy as np

from ocr import PaddleOCR, parse_args
from ppocr.utils.network import is_link, download_with_progressbar
from ppocr.utils.utility import get_image_file_list

from ppocr.utils.logging import get_logger

logger = get_logger()


def layout_xlmv2(input_file: Optional[str] = None, det: bool = True, rec: bool = False):
    """
    Implement paddle text detection and recognition based on value in the parameters.
    params:
        input_file = File given by the user. None or string of path to the file
        det = Default True, Detection must be True.
        rec = Default False.
    """
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
        # img_name = os.path.basename(img_path).split('.')[0]
        logger.info('{}{}{}'.format('*' * 10, img_path, '*' * 10))
        result = engine.ocr(img_path,
                            det=det,
                            rec=rec,
                            cls=args.use_angle_cls,
                            bin=args.binarize,
                            inv=args.invert,
                            alpha_color=args.alphacolor)


        print(result)

        # if rec:
        #     for page in result:
        #         bboxes = []
        #         img = page[0]
        #         det_rec_results = page[1]
        #
        #         for data_in_each_page in det_rec_results:
        #             bbox = data_in_each_page[0]
        #             rec_text = data_in_each_page[1]
        #             print(rec_text)
        #             bboxes.append(bbox)
        #         visualize_bboxes(img=img, bboxes=bboxes, rec=rec)

        # else:
        #     visualize_bboxes(img=img, bboxes=bboxes, rec=rec)


def visualize_bboxes(img: np.ndarray, bboxes: List, rec: bool = False):
    """
        Draws the detected bounding boxes round the detected text.

    params:
        img_path = path to the original image (each page)
        bboxes = list of all of the bounding boxes of the current page.
    """
    # img = cv2.imread(img_path)
    if not rec:
        bboxes = bboxes[0]
    for bbox in bboxes:
        l, t = bbox[0]
        r, b = bbox[2]

        img = cv2.rectangle(img, (int(l), int(t)), (int(r), int(b)), [255, 0, 0], 2)

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)
    cv2.waitKey(0)


if __name__ == "__main__":
    # image_path = "/home/vertexaiml/Downloads/ocr_test_image/nepal.png"
    image_path = "/home/vertexaiml/Downloads/ocr_test_image/4 page.jpg"
    # image_path = "/home/vertexaiml/Documents/20231009_205723.jpg"
    # image_path = "/home/vertexaiml/Downloads/ocr_test_image/blank_white.jpg"
    # file_path = "/home/vertexaiml/Downloads/Vertex_It/Poc_Sample/Bank_Of_America/Bank of America.pdf"

    layout_xlmv2(input_file=image_path, det=True, rec=True)
    # paddle_det_and_rec(input_file=file_path, det=True, rec=True)
