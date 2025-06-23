import time
from models.yolo import yolov10
from models import gemini

class BoundingBox(object):
    x1: int
    x2: int
    y1: int
    y2: int

class Frame(object):
    def __init__(self, image_path: str):
        self.image_path = image_path

    def run_predict(self):
        """
        Extract all text content from image
        """
        
        message = '''
        Please extract all textual information from the uploaded image.
        Return the result in the following JSON format:
        {
            "text": "extracted text information from the image in Vietnamese"
        }
        '''

        start_time = time.time()
        response = gemini.prompting(message=message, image_paths=[self.image_path])

        print("--- %s seconds ---" % (time.time() - start_time))
        return response