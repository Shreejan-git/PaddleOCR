#
#
#
#
#
# class PPStructure(StructureSystem):
#     def __init__(self, **kwargs):
#         params = parse_args(mMain=False)
#         params.__dict__.update(**kwargs)
#         assert params.structure_version in SUPPORT_STRUCTURE_MODEL_VERSION, "structure_version must in {}, but get {}".format(
#             SUPPORT_STRUCTURE_MODEL_VERSION, params.structure_version)
#         params.use_gpu = check_gpu(params.use_gpu)
#         params.mode = 'structure'
#
#         if not params.show_log:
#             logger.setLevel(logging.INFO)
#         lang, det_lang = parse_lang(params.lang)
#         if lang == 'ch':
#             table_lang = 'ch'
#         else:
#             table_lang = 'en'
#         if params.structure_version == 'PP-Structure':
#             params.merge_no_span_structure = False
#
#         # init model dir
#         det_model_config = get_model_config('OCR', params.ocr_version, 'det',
#                                             det_lang)
#         params.det_model_dir, det_url = confirm_model_dir_url(
#             params.det_model_dir,
#             os.path.join(BASE_DIR, 'whl', 'det', det_lang),
#             det_model_config['url'])
#         rec_model_config = get_model_config('OCR', params.ocr_version, 'rec',
#                                             lang)
#         params.rec_model_dir, rec_url = confirm_model_dir_url(
#             params.rec_model_dir,
#             os.path.join(BASE_DIR, 'whl', 'rec', lang), rec_model_config['url'])
#         table_model_config = get_model_config(
#             'STRUCTURE', params.structure_version, 'table', table_lang)
#         params.table_model_dir, table_url = confirm_model_dir_url(
#             params.table_model_dir,
#             os.path.join(BASE_DIR, 'whl', 'table'), table_model_config['url'])
#         layout_model_config = get_model_config(
#             'STRUCTURE', params.structure_version, 'layout', lang)
#         params.layout_model_dir, layout_url = confirm_model_dir_url(
#             params.layout_model_dir,
#             os.path.join(BASE_DIR, 'whl', 'layout'), layout_model_config['url'])
#         # download model
#         maybe_download(params.det_model_dir, det_url)
#         maybe_download(params.rec_model_dir, rec_url)
#         maybe_download(params.table_model_dir, table_url)
#         maybe_download(params.layout_model_dir, layout_url)
#
#         if params.rec_char_dict_path is None:
#             params.rec_char_dict_path = str(
#                 Path(__file__).parent / rec_model_config['dict_path'])
#         if params.table_char_dict_path is None:
#             params.table_char_dict_path = str(
#                 Path(__file__).parent / table_model_config['dict_path'])
#         if params.layout_dict_path is None:
#             params.layout_dict_path = str(
#                 Path(__file__).parent / layout_model_config['dict_path'])
#         logger.debug(params)
#         super().__init__(params)
#
#     def __call__(self, img, return_ocr_result_in_table=False, img_idx=0):
#         img = check_img(img)
#         res, _ = super().__call__(
#             img, return_ocr_result_in_table, img_idx=img_idx)
#         return res
#
#
# def main():
#     # for cmd
#     args = parse_args(mMain=True)
#     image_dir = args.image_dir
#     if is_link(image_dir):
#         download_with_progressbar(image_dir, 'tmp.jpg')
#         image_file_list = ['tmp.jpg']
#     else:
#         image_file_list = get_image_file_list(args.image_dir)
#     if len(image_file_list) == 0:
#         logger.error('no images find in {}'.format(args.image_dir))
#         return
#     if args.type == 'ocr':
#         engine = PaddleOCR(**(args.__dict__))
#     elif args.type == 'structure':
#         engine = PPStructure(**(args.__dict__))
#     else:
#         raise NotImplementedError
#
#     for img_path in image_file_list:
#         img_name = os.path.basename(img_path).split('.')[0]
#         logger.info('{}{}{}'.format('*' * 10, img_path, '*' * 10))
#         if args.type == 'ocr':
#             result = engine.ocr(img_path,
#                                 det=args.det,
#                                 rec=args.rec,
#                                 cls=args.use_angle_cls,
#                                 bin=args.binarize,
#                                 inv=args.invert,
#                                 alpha_color=args.alphacolor)
#             if result is not None:
#                 for idx in range(len(result)):
#                     res = result[idx]
#                     for line in res:
#                         logger.info(line)
#         elif args.type == 'structure':
#             img, flag_gif, flag_pdf = check_and_read(img_path)
#             if not flag_gif and not flag_pdf:
#                 img = cv2.imread(img_path)
#
#             if args.recovery and args.use_pdf2docx_api and flag_pdf:
#                 from pdf2docx.converter import Converter
#                 docx_file = os.path.join(args.output,
#                                          '{}.docx'.format(img_name))
#                 cv = Converter(img_path)
#                 cv.convert(docx_file)
#                 cv.close()
#                 logger.info('docx save to {}'.format(docx_file))
#                 continue
#
#             if not flag_pdf:
#                 if img is None:
#                     logger.error("error in loading image:{}".format(img_path))
#                     continue
#                 img_paths = [[img_path, img]]
#             else:
#                 img_paths = []
#                 for index, pdf_img in enumerate(img):
#                     os.makedirs(
#                         os.path.join(args.output, img_name), exist_ok=True)
#                     pdf_img_path = os.path.join(
#                         args.output, img_name,
#                         img_name + '_' + str(index) + '.jpg')
#                     cv2.imwrite(pdf_img_path, pdf_img)
#                     img_paths.append([pdf_img_path, pdf_img])
#
#             all_res = []
#             for index, (new_img_path, img) in enumerate(img_paths):
#                 logger.info('processing {}/{} page:'.format(index + 1,
#                                                             len(img_paths)))
#                 new_img_name = os.path.basename(new_img_path).split('.')[0]
#                 result = engine(img, img_idx=index)
#                 save_structure_res(result, args.output, img_name, index)
#
#                 if args.recovery and result != []:
#                     from copy import deepcopy
#                     from ppstructure.recovery.recovery_to_doc import sorted_layout_boxes
#                     h, w, _ = img.shape
#                     result_cp = deepcopy(result)
#                     result_sorted = sorted_layout_boxes(result_cp, w)
#                     all_res += result_sorted
#
#             if args.recovery and all_res != []:
#                 try:
#                     from ppstructure.recovery.recovery_to_doc import convert_info_docx
#                     convert_info_docx(img, all_res, args.output, img_name)
#                 except Exception as ex:
#                     logger.error(
#                         "error in layout recovery image:{}, err msg: {}".format(
#                             img_name, ex))
#                     continue
#
#             for item in all_res:
#                 item.pop('img')
#                 item.pop('res')
#                 logger.info(item)
#             logger.info('result save to {}'.format(args.output))
