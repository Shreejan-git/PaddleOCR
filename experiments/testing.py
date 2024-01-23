import logging
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VertexOCR:

    def __init__(self):
        self.args = parse_args(mMain=True)

    def process_image(self, img):
        try:
            logging.debug(f"Processing image: {img}")
            engine = PPStructure(**self.args.__dict__)
            result = engine(img)
            logging.debug(f"Processing complete for image: {img}")
            return result
        except Exception as e:
            logging.error(f"Error processing image: {e}", exc_info=True)
            return None

    @contextmanager
    def temporary_file(self, input_file):
        try:
            download_with_progressbar(input_file, 'tmp.jpg')
            yield 'tmp.jpg'
        finally:
            os.remove('tmp.jpg')

    def process_files(self, file_list, max_workers):
        results = []
        for input_file in file_list:
            img_name = os.path.basename(input_file).split('.')[0]
            logging.info(f"Processing file: {input_file} (Image Name: {img_name})")

            total_page_count, imgs, flag_gif, flag_pdf = check_and_read(input_file)
            if not flag_gif and not flag_pdf:
                imgs = [cv2.imread(input_file)]

            if imgs:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    logging.info("Starting multiprocessing for image processing")
                    try:
                        processed = list(executor.map(self.process_image, imgs))
                        results.extend(processed)
                    except Exception as e:
                        logging.error(f"Error during multiprocessing: {e}", exc_info=True)
        return results

    def extract_layout(self, input_file: str, visualize=False, max_workers=None):
        logging.info(f"Starting layout extraction for: {input_file}")

        if max_workers is None:
            max_workers = os.cpu_count() or 1

        file_list = self.get_file_list(input_file)
        if not file_list:
            logging.error(f"No images found in {input_file}")
            return

        results = self.process_files(file_list, max_workers)
        results = [result for result in results if result is not None]

        logging.info(f"Layout extraction completed for: {input_file}")
        return results

    def get_file_list(self, input_file):
        if is_link(input_file):
            with self.temporary_file(input_file) as temp_file:
                return [temp_file]
        else:
            return get_image_file_list(input_file)

# Rest of your code...
