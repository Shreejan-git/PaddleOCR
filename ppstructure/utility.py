import os
import random
import ast
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tools.infer.utility import draw_ocr_box_txt, str2bool, str2int_tuple, init_args as infer_args
import logging
current_file_absolute_dir_path = os.path.dirname(__file__)


def init_args():
    parser = infer_args()

    # params for output
    parser.add_argument("--output", type=str, default='./output')
    # params for table structure
    parser.add_argument("--table_max_len", type=int, default=488)
    parser.add_argument("--table_algorithm", type=str, default='TableAttn')
    parser.add_argument("--table_model_dir", type=str)
    parser.add_argument(
        "--merge_no_span_structure", type=str2bool, default=True)
    parser.add_argument(
        "--table_char_dict_path",
        type=str,
        default="../ppocr/utils/dict/table_structure_dict_ch.txt")
    # params for layout
    parser.add_argument("--layout_model_dir", type=str)

    layout_publaynet_dict_path = os.path.join(current_file_absolute_dir_path, '..', 'ppocr', 'utils', 'dict',
                                              'layout_dict', 'layout_publaynet_dict.txt')
    if os.path.exists(layout_publaynet_dict_path):
        parser.add_argument(
            "--layout_dict_path",
            type=str,
            # default="../ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt")
            default=layout_publaynet_dict_path)
    else:
        logging.info(f'INFO [Could not find en_dict_path in {__file__} file]')

    parser.add_argument(
        "--layout_score_threshold",
        type=float,
        default=0.5,
        help="Threshold of score.")
    parser.add_argument(
        "--layout_nms_threshold",
        type=float,
        default=0.5,
        help="Threshold of nms.")

    # params for kie
    # parser.add_argument("--kie_algorithm", type=str, default='LayoutXLM')
    # # parser.add_argument("--ser_model_dir", type=str)
    # parser.add_argument("--ser_model_dir", type=str, default="/home/vertexaiml/Documents/PaddleOCR"
    #                                                          "/pretrained_model_kie/ser_vi_layoutxlm_xfund_infer")
    # # parser.add_argument("--re_model_dir", type=str)
    # parser.add_argument("--re_model_dir", type=str, default="/home/vertexaiml/Documents/PaddleOCR"
    #                                                         "/pretrained_model_kie/re_vi_layoutxlm_xfund_infer")
    # # parser.add_argument("--use_visual_backbone", type=str2bool, default=True)  # True hunuhudaina error auxa.
    # parser.add_argument("--use_visual_backbone", type=str2bool, default=False)
    # parser.add_argument(
    #     "--ser_dict_path",
    #     type=str,
    #     # default="../train_data/XFUND/class_list_xfun.txt")
    #     default="/home/vertexaiml/Desktop/train_data/XFUND/class_list_xfun.txt")
    # need to be None or tb-yx

    parser.add_argument("--ocr_order_method", type=str, default=None)
    # params for inference
    parser.add_argument(
        "--mode",
        type=str,
        choices=['structure', 'kie'],
        default='structure',
        help='structure and kie is supported')
    parser.add_argument(
        "--image_orientation",
        type=bool,
        default=False,
        help='Whether to enable image orientation recognition')
    parser.add_argument(
        "--layout",
        type=str2bool,
        default=True,
        help='Whether to enable layout analysis')
    parser.add_argument(
        "--table",
        type=str2bool,
        default=True,
        help='In the forward, whether the table area uses table recognition')
    parser.add_argument(
        "--ocr",
        type=str2bool,
        default=True,
        help='In the forward, whether the non-table area is recognition by ocr')
    # param for recovery
    parser.add_argument(
        "--recovery",
        type=str2bool,
        default=False,
        help='Whether to enable layout of recovery')
    parser.add_argument(
        "--use_pdf2docx_api",
        type=str2bool,
        default=False,
        help='Whether to use pdf2docx api')
    parser.add_argument(
        "--invert",
        type=str2bool,
        default=False,
        help='Whether to invert image before processing')
    parser.add_argument(
        "--binarize",
        type=str2bool,
        default=False,
        help='Whether to threshold binarize image before processing')
    parser.add_argument(
        "--alphacolor",
        type=str2int_tuple,
        default=(255, 255, 255),
        help='Replacement color for the alpha channel, if the latter is present; R,G,B integers')

    return parser


def parse_args():
    parser = init_args()
    return parser.parse_args()


def draw_structure_result(image, result, font_path):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    boxes, txts, scores = [], [], []

    img_layout = image.copy()
    draw_layout = ImageDraw.Draw(img_layout)
    text_color = (255, 255, 255)
    text_background_color = (80, 127, 255)
    catid2color = {}
    font_size = 15
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    for region in result:
        if region['type'] not in catid2color:
            box_color = (random.randint(0, 255), random.randint(0, 255),
                         random.randint(0, 255))
            catid2color[region['type']] = box_color
        else:
            box_color = catid2color[region['type']]
        box_layout = region['bbox']
        draw_layout.rectangle(
            [(box_layout[0], box_layout[1]), (box_layout[2], box_layout[3])],
            outline=box_color,
            width=3)
        left, top, right, bottom = font.getbbox(region['type'])
        text_w, text_h = right - left, bottom - top
        draw_layout.rectangle(
            [(box_layout[0], box_layout[1]),
             (box_layout[0] + text_w, box_layout[1] + text_h)],
            fill=text_background_color)
        draw_layout.text(
            (box_layout[0], box_layout[1]),
            region['type'],
            fill=text_color,
            font=font)

        if region['type'] == 'table':
            pass
        else:
            for text_result in region['res']:
                boxes.append(np.array(text_result['text_region']))
                txts.append(text_result['text'])
                scores.append(text_result['confidence'])

    im_show = draw_ocr_box_txt(
        img_layout, boxes, txts, scores, font_path=font_path, drop_score=0)
    return im_show
