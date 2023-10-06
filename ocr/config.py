import importlib
import os
import sys
import paddle

# from ocr.paddle_ocr import PaddleOCR
# from ocr.paddle_structure import PPStructure

# from ppocr.utils.logging import get_logger
# from ppocr.utils.network import maybe_download, download_with_progressbar, is_link, confirm_model_dir_url
from tools.infer.utility import draw_ocr, str2bool, check_gpu
from ppstructure.utility import init_args, draw_structure_result
# from ppstructure.predict_system import StructureSystem, save_structure_res, to_excel

from ppocr.utils.logging import get_logger
logger = get_logger()

__dir__ = os.path.dirname(__file__)
sys.path.append(os.path.join(__dir__, ''))


ppocr = importlib.import_module('ppocr', 'paddleocr')
ppstructure = importlib.import_module('ppstructure', 'paddleocr')

# logger = get_logger()
# __all__ = [
#     'PaddleOCR', 'PPStructure', 'draw_ocr', 'draw_structure_result',
#     'save_structure_res', 'download_with_progressbar', 'to_excel'
# ]

SUPPORT_DET_MODEL = ['DB', 'DB++']
VERSION = '2.7.0.3'
SUPPORT_REC_MODEL = ['CRNN', 'SVTR_LCNet']
BASE_DIR = os.path.expanduser("~/.paddleocr/")

DEFAULT_OCR_MODEL_VERSION = 'PP-OCRv4'
SUPPORT_OCR_MODEL_VERSION = ['PP-OCR', 'PP-OCRv2', 'PP-OCRv3', 'PP-OCRv4']
DEFAULT_STRUCTURE_MODEL_VERSION = 'PP-StructureV2'
SUPPORT_STRUCTURE_MODEL_VERSION = ['PP-Structure', 'PP-StructureV2']
MODEL_URLS = {
    'OCR': {
        'PP-OCRv4': {
            'det': {
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar',
                },
                'en': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar',
                },
                'ml': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar'
                }
            },
            'rec': {
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar',
                    'dict_path': './ppocr/utils/ppocr_keys_v1.txt'
                },
                'en': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar',
                    'dict_path': './ppocr/utils/en_dict.txt'
                },
                'korean': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/korean_PP-OCRv4_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/korean_dict.txt'
                },
                'japan': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/japan_PP-OCRv4_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/japan_dict.txt'
                },
                'chinese_cht': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/chinese_cht_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/chinese_cht_dict.txt'
                },
                'ta': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/ta_PP-OCRv4_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/ta_dict.txt'
                },
                'te': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/te_PP-OCRv4_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/te_dict.txt'
                },
                'ka': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/ka_PP-OCRv4_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/ka_dict.txt'
                },
                'latin': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/latin_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/latin_dict.txt'
                },
                'arabic': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/arabic_PP-OCRv4_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/arabic_dict.txt'
                },
                'cyrillic': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/cyrillic_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/cyrillic_dict.txt'
                },
                'devanagari': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/devanagari_PP-OCRv4_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/devanagari_dict.txt'
                },
            },
            'cls': {
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar',
                }
            },
        },
        'PP-OCRv3': {
            'det': {
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar',
                },
                'en': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar',
                },
                'ml': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar'
                }
            },
            'rec': {
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/ppocr_keys_v1.txt'
                },
                'en': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/en_dict.txt'
                },
                'korean': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/korean_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/korean_dict.txt'
                },
                'japan': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/japan_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/japan_dict.txt'
                },
                'chinese_cht': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/chinese_cht_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/chinese_cht_dict.txt'
                },
                'ta': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ta_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/ta_dict.txt'
                },
                'te': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/te_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/te_dict.txt'
                },
                'ka': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ka_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/ka_dict.txt'
                },
                'latin': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/latin_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/latin_dict.txt'
                },
                'arabic': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/arabic_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/arabic_dict.txt'
                },
                'cyrillic': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/cyrillic_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/cyrillic_dict.txt'
                },
                'devanagari': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/devanagari_PP-OCRv3_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/devanagari_dict.txt'
                },
            },
            'cls': {
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar',
                }
            },
        },
        'PP-OCRv2': {
            'det': {
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar',
                },
            },
            'rec': {
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar',
                    'dict_path': './ppocr/utils/ppocr_keys_v1.txt'
                }
            },
            'cls': {
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar',
                }
            },
        },
        'PP-OCR': {
            'det': {
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar',
                },
                'en': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_ppocr_mobile_v2.0_det_infer.tar',
                },
                'structure': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_det_infer.tar'
                }
            },
            'rec': {
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/ppocr_keys_v1.txt'
                },
                'en': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/en_dict.txt'
                },
                'french': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/french_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/french_dict.txt'
                },
                'german': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/german_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/german_dict.txt'
                },
                'korean': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/korean_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/korean_dict.txt'
                },
                'japan': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/japan_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/japan_dict.txt'
                },
                'chinese_cht': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/chinese_cht_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/chinese_cht_dict.txt'
                },
                'ta': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ta_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/ta_dict.txt'
                },
                'te': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/te_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/te_dict.txt'
                },
                'ka': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ka_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/ka_dict.txt'
                },
                'latin': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/latin_ppocr_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/latin_dict.txt'
                },
                'arabic': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/arabic_ppocr_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/arabic_dict.txt'
                },
                'cyrillic': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/cyrillic_ppocr_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/cyrillic_dict.txt'
                },
                'devanagari': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/devanagari_ppocr_mobile_v2.0_rec_infer.tar',
                    'dict_path': './ppocr/utils/dict/devanagari_dict.txt'
                },
                'structure': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_rec_infer.tar',
                    'dict_path': 'ppocr/utils/dict/table_dict.txt'
                }
            },
            'cls': {
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar',
                }
            },
        }
    },
    'STRUCTURE': {
        'PP-Structure': {
            'table': {
                'en': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar',
                    'dict_path': 'ppocr/utils/dict/table_structure_dict.txt'
                }
            }
        },
        'PP-StructureV2': {
            'table': {
                'en': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_infer.tar',
                    'dict_path': 'ppocr/utils/dict/table_structure_dict.txt'
                },
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/ch_ppstructure_mobile_v2.0_SLANet_infer.tar',
                    'dict_path': 'ppocr/utils/dict/table_structure_dict_ch.txt'
                }
            },
            'layout': {
                'en': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_infer.tar',
                    'dict_path':
                        'ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt'
                },
                'ch': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_cdla_infer.tar',
                    'dict_path':
                        'ppocr/utils/dict/layout_dict/layout_cdla_dict.txt'
                }
            }
        }
    }
}


def parse_args(mMain=True):
    import argparse
    parser = init_args()
    parser.add_help = mMain
    # parser.add_argument("--lang", type=str, default='en')
    parser.add_argument("--det", type=str2bool, default=True)
    parser.add_argument("--rec", type=str2bool, default=False)
    parser.add_argument("--type", type=str, default='ocr')
    parser.add_argument(
        "--ocr_version",
        type=str,
        choices=SUPPORT_OCR_MODEL_VERSION,
        default='PP-OCRv4',
        help='OCR Model version, the current model support list is as follows: '
             '1. PP-OCRv4/v3 Support Chinese and English detection and recognition model, and direction classifier model'
             '2. PP-OCRv2 Support Chinese detection and recognition model. '
             '3. PP-OCR support Chinese detection, recognition and direction classifier and multilingual recognition model.'
    )
    parser.add_argument(
        "--structure_version",
        type=str,
        choices=SUPPORT_STRUCTURE_MODEL_VERSION,
        default='PP-StructureV2',
        help='Model version, the current model support list is as follows:'
             ' 1. PP-Structure Support en table structure model.'
             ' 2. PP-StructureV2 Support ch and en table structure model.')

    for action in parser._actions:
        if action.dest in [
            'rec_char_dict_path', 'table_char_dict_path', 'layout_dict_path'
        ]:
            action.default = None
    if mMain:
        return parser.parse_args()
    else:
        inference_args_dict = {}
        for action in parser._actions:
            inference_args_dict[action.dest] = action.default
        return argparse.Namespace(**inference_args_dict)


def parse_lang(lang):
    latin_lang = [
        'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',
        'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
        'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
        'sw', 'tl', 'tr', 'uz', 'vi', 'french', 'german'
    ]
    arabic_lang = ['ar', 'fa', 'ug', 'ur']
    cyrillic_lang = [
        'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd', 'ava',
        'dar', 'inh', 'che', 'lbe', 'lez', 'tab'
    ]
    devanagari_lang = [
        'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom',
        'sa', 'bgc'
    ]
    if lang in latin_lang:
        lang = "latin"
    elif lang in arabic_lang:
        lang = "arabic"
    elif lang in cyrillic_lang:
        lang = "cyrillic"
    elif lang in devanagari_lang:
        lang = "devanagari"
    assert lang in MODEL_URLS['OCR'][DEFAULT_OCR_MODEL_VERSION][
        'rec'], 'param lang must in {}, but got {}'.format(
        MODEL_URLS['OCR'][DEFAULT_OCR_MODEL_VERSION]['rec'].keys(), lang)
    if lang == "ch":
        det_lang = "ch"
    elif lang == 'structure':
        det_lang = 'structure'
    elif lang in ["en", "latin"]:
        det_lang = "en"
    else:
        det_lang = "ml"
    return lang, det_lang


def get_model_config(type, version, model_type, lang):
    if type == 'OCR':
        DEFAULT_MODEL_VERSION = DEFAULT_OCR_MODEL_VERSION
    elif type == 'STRUCTURE':
        DEFAULT_MODEL_VERSION = DEFAULT_STRUCTURE_MODEL_VERSION
    else:
        raise NotImplementedError

    model_urls = MODEL_URLS[type]
    if version not in model_urls:
        version = DEFAULT_MODEL_VERSION
    if model_type not in model_urls[version]:
        if model_type in model_urls[DEFAULT_MODEL_VERSION]:
            version = DEFAULT_MODEL_VERSION
        else:
            logger.error('{} models is not support, we only support {}'.format(
                model_type, model_urls[DEFAULT_MODEL_VERSION].keys()))
            sys.exit(-1)

    if lang not in model_urls[version][model_type]:
        if lang in model_urls[DEFAULT_MODEL_VERSION][model_type]:
            version = DEFAULT_MODEL_VERSION
        else:
            logger.error(
                'lang {} is not support, we only support {} for {} models'.
                format(lang, model_urls[DEFAULT_MODEL_VERSION][model_type].keys(
                ), model_type))
            sys.exit(-1)
    return model_urls[version][model_type][lang]
