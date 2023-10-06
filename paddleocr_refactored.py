# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import importlib

__dir__ = os.path.dirname(__file__)

import paddle

sys.path.append(os.path.join(__dir__, ''))

import cv2
import logging
import numpy as np
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image


def _import_file(module_name, file_path, make_importable=False):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if make_importable:
        sys.modules[module_name] = module
    return module


tools = _import_file(
    'tools', os.path.join(__dir__, 'tools/__init__.py'), make_importable=True)
ppocr = importlib.import_module('ppocr', 'paddleocr')
ppstructure = importlib.import_module('ppstructure', 'paddleocr')
from ppocr.utils.logging import get_logger
from tools.infer import predict_system
from ppocr.utils.utility import check_and_read, get_image_file_list, alpha_to_color, binarize_img
from ppocr.utils.network import maybe_download, download_with_progressbar, is_link, confirm_model_dir_url
from tools.infer.utility import draw_ocr, str2bool, check_gpu
from ppstructure.utility import init_args, draw_structure_result
from ppstructure.predict_system import StructureSystem, save_structure_res, to_excel






