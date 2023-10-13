import os
import random
import cv2
from ocr import parse_args, PPStructure
from ppocr.utils.network import is_link, download_with_progressbar
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.utils.logging import get_logger

logger = get_logger()


def paddle_layout_table_extraction(file_path:str):
    args = parse_args(mMain=True)
    if is_link(file_path):
        download_with_progressbar(file_path, 'tmp.jpg')
        image_file_list = ['tmp.jpg']
    else:
        image_file_list = get_image_file_list(file_path)
    if len(image_file_list) == 0:
        logger.error('no images find in {}'.format(file_path))
        return
    # engine = PPStructure(lang='che', **args.__dict__)
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
    # file_path = "/home/vertexaiml/Downloads/ocr_test_image/4 page.jpg"
    # file_path = "/home/vertexaiml/Downloads/ocr_test_image/test_invoice.png"
    # file_path = "/home/vertexaiml/Downloads/ocr_test_image/test_invoice.png"
    file_path = "/home/vertexaiml/Downloads/Vertex_It/Poc_Sample/Bank_Of_America/Bank of America.pdf"
    paddle_layout_table_extraction(file_path=file_path)
