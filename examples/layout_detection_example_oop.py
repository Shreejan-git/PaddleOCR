import os
import pymongo
from concurrent.futures import ProcessPoolExecutor
import tempfile
from ocr import parse_args, PPStructure
from ppocr.utils.utility import check_and_read
from ppocr.utils.logging import get_logger
from ppocr.utils.network import download_file_from_s3
from contextlib import contextmanager

logger = get_logger()


class VertexOCR:

    def __init__(self):
        self.args = parse_args(mMain=True)

    @contextmanager
    def temporary_file(self, uuid, file_name: str, input_file: str):
        suffix = input_file.split('.')[-1]
        fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=str(uuid) + file_name, dir='/tmp')
        os.close(fd)
        try:
            download_file_from_s3(bucket_name='vertex-clients-bucket', s3_file_key='12345', local_file_path=temp_path)
            yield temp_path
        finally:
            os.remove(temp_path)

    def process_image(self, img):

        try:
            engine = PPStructure(**self.args.__dict__)  # Use appropriate arguments
            return engine(img)
        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            return None

    def process_files(self, input_file: str, max_workers: int):
        """
         If the input_file is a Pdf file each page is converted into images, else passes image to
         initialized multiprocessor and returns the result.

                 Parameters:
                         input_file (str): A decimal integer
                         max_workers (int): Another decimal integer

                 Returns:
        """
        img_name = os.path.basename(input_file).split('.')[0]
        logger.info(f"Processing file: {input_file} (Image Name: {img_name})")

        total_page_count, processed_input, is_image, is_pdf, is_none = check_and_read(input_file)

        if is_pdf:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                logger.info("Starting multiprocessing for Layout Extraction")
                try:
                    processed = list(executor.map(self.process_image, processed_input))
                    results = [result for result in processed if result is not None]

                    return results

                except Exception as e:
                    logger.error(f"Error during multiprocessing: {e}", exc_info=True)

        if is_image:
            processed = self.process_image(img=processed_input)
            results = [processed]
            return results

        if is_none:
            '''
            If we get a file other that pdf and supported 3 images types then this section must be triggered.
            '''
            print("raise exception")
            return 0

    def extract_layout(self, input_file: str, visualize=False, max_workers=None):

        if max_workers is None:
            max_workers = os.cpu_count() or 1
        '''
        file_uuid = input_file['id']
        file_path = input_file['s3_file_path']
        file_name = input_file['file_name']
        
        with self.temporary_file(uuid=file_uuid, file_name=file_name, input_file=file_path) as temp_file:
            results = self.process_files(input_file=temp_file, max_workers=max_workers)

            return results
        '''
        client = pymongo.MongoClient("mongodb+srv://sangam:12345@cluster0.qtchyof.mongodb.net/")
        db = client['ocr']
        collection = db['layout']
        dictionary = {'_id': 5,
                      'progress_status': 75,
                      's3_path': 'link_here'}

        collection.insert_one(dictionary)

        results = self.process_files(input_file=input_file, max_workers=max_workers)
        logger.info(f"Layout extraction completed for: {input_file}")

        # when we get to this point, the layout extraction portion is completed. So, need to update in the
        # MONGODB calculating the percentage
        print("******************************************")
        print(type(results))
        print(len(results))
        print(type(results[0]))
        print(type(results[0][0]))
        print(results[0][0]['res'])
        # print(results)
        # print(results)
        collection.update_one({"_id": 5}, {"$set": {"progress_status": 100}})

        return results

    def post_process_layout_extraction(self, layout_extraction_result: list):
        """
        Takes the output of layout extraction as an input and returns the filtered content from each file
        """

        # IMPORTANT !!!
        # PDF ho vane matrai multiprocessing lagaunu parxa or input nai multiple images ho vane matrai.
        # single image matrai input ayeko xa vaye 1 core use garnu parne xa.

        # img_paths: List[List[Union[str, np.ndarray]]] = []  # yo aile chaiye ko chai xaina.
        # for index, pdf_img in enumerate(imgs, start=1):
        #     os.makedirs(
        #         os.path.join(self.args.output, img_name), exist_ok=True)
        #     pdf_img_path = os.path.join(
        #         self.args.output, img_name,
        #         img_name + '_' + str(index) + '.jpg')
        #     # cv2.imwrite(pdf_img_path, pdf_img)
        #     img_paths.append([pdf_img_path, pdf_img])

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
    import time

    file_path = "/home/sangam/Downloads/Vertex_It/Poc_Sample/Bank_Of_America/Bank of America.pdf"
    # file_path = "/home/vertexaiml/Downloads/Vertex_It/Poc_Sample/Wellsfargo/wellsfargo_pdf"
    # paddle_layout_table_extraction(file_path=file_path, visualize=True)
    vertex_ocr = VertexOCR()
    s = time.time()
    vertex_ocr.extract_layout(input_file=file_path)
    print((time.time() - s) / 60)
