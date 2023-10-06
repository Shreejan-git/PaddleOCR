import os
import cv2
from ocr import PaddleOCR
from ocr.config import parse_args
from ppocr.utils.network import is_link, download_with_progressbar
from ppocr.utils.utility import get_image_file_list

from ppocr.utils.logging import get_logger
logger = get_logger()


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
            for rec_result in result:
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
    # image_path = "/home/vertexaiml/Downloads/ocr_test_image/nepal.png"
    image_path = "/home/vertexaiml/Downloads/4 page.jpg"
    # image_path = "/home/vertexaiml/Downloads/ocr_test_image/blank_white.jpg"
    paddle_det_and_rec(input_file=image_path, det=True, rec=False)
