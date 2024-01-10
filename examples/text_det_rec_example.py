from typing import Optional, List

import cv2
import numpy as np

from ocr import PaddleOCR, parse_args
from ppocr.utils.network import is_link, download_with_progressbar
from ppocr.utils.utility import get_image_file_list

from ppocr.utils.logging import get_logger

logger = get_logger()
import os
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from getpass import getpass

import json


def paddle_det_and_rec(input_file: Optional[str] = None, det: bool = True, rec: bool = False):
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

        if rec:
            for page in result:
                bboxes = []
                img = page[0]
                det_rec_results = page[1]
                all_recognized_text = []
                for data_in_each_page in det_rec_results:
                    bbox = data_in_each_page[0]
                    rec_text = data_in_each_page[1]
                    # print(rec_text)
                    all_recognized_text.append(rec_text[0])
                    bboxes.append(bbox)
                visualize_bboxes(img=img, bboxes=bboxes, rec=rec)
                # print(all_recognized_text)
                # else:
                #     visualize_bboxes(img=img, bboxes=bboxes, rec=rec)
                # print(all_recognized_text)
                return all_recognized_text


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


def llm_component(img_link):
    prompt_text = paddle_det_and_rec(input_file=img_link, det=True, rec=True)
    # prompt_text = """
    # You are an expert admin who will extract core information from documents
    #
    # {recognized_text}
    #
    # Above is the content; please try to extract all data points from the content above and export and
    # return in a JSON array format only. Do not auto generate the information. Only return what
    # there is in the document
    #
    # """

    prompt_template = PromptTemplate(
        input_variables=['recognized_text'],

        template="""   
        {recognized_text}
        from above content try to extract the following data:
        1) Extract the details of the bank such as name, contact number, website and address (postbox, city, state, country,
            zip)
        2) Details of the client. Such as name, account no, statement start date, statement end date, account type, currency
        type and address (postbox, city, state, country, zip)
        
        Format the response as JSON object with the following keys:
        "bankDetails", "clientDetails"
        Do not auto generate the information. Only return what is there in document or else
        return an empty string.
 """
    )
    llm = OpenAI(openai_api_key="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")
    llm_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

    # sequential_chain = SimpleSequentialChain(chains=[llm_chain], input_variables=['text'], verbose=True)

    response = llm_chain.run(prompt_text)
    print(response)
    # r = type(response)
    # print(r)

    # returning_file = json.dumps(response)
    # print(returning_file)

    # return returning_file


def file_list():
    # image_path = "/home/vertexaiml/Downloads/ocr_test_image/nepal.png"
    # image_path = "/home/vertexaiml/Downloads/ocr_test_image/4 page.jpg"
    # image_path = "/home/vertexaiml/Downloads/ocr_test_image/testing.jpg"
    # image_path = "/home/vertexaiml/Downloads/ocr_test_image/sample_Invoice.png"
    # image_path = "/home/vertexaiml/Downloads/ocr_test_image/unstructured.png"
    # image_path = "/home/vertexaiml/Downloads/ocr_test_image/unstructured_second.png"
    # image_path = "/home/vertexaiml/Downloads/ocr_test_image/unstructured_third.png"
    # image_path = "/home/vertexaiml/Downloads/ocr_test_image/unstructured_forth.png"
    # image_path = "/home/vertexaiml/Downloads/ocr_test_image/second_table_data.png"
    image_path = "/home/vertexaiml/Downloads/ocr_test_image/tabular_small.jpg"
    # image_path = "/home/vertexaiml/Downloads/Vertex_It/Poc_Sample/Bank_Of_America/images/5.png"


if __name__ == "__main__":
    # paddle_det_and_rec(input_file=image_path, det=True, rec=True)
    image_path = "/home/vertexaiml/Downloads/Vertex_It/Poc_Sample/Bank_Of_America/images/1cropped.png"
    paddle_det_and_rec(input_file=image_path, det=True, rec=True)
    # llm_component(image_path)
    # print(result)
    # print(os.environ.get('OPENAI_API_KEY'))
