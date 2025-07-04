from concurrent.futures import ThreadPoolExecutor
import cv2
import time
import tempfile
from models.yolo import yolov10
from models import gemini, paddle


class Frame(object):
    def __init__(self, image_path: str):
        self.image_path = image_path

    def gemini_predict(self):
        message = """
        Please extract all textual information from the uploaded image.
        Return the result in the following JSON format:
        {
            "text": "extracted text information from the image in Vietnamese"
        }
        """

        response = gemini.prompting(message=message, image_paths=[self.image_path])
        return response

    def paddle_predict(self):
        response = paddle.predict(image_path=self.image_path)
        return response

    def run_predict(self):
        """
        Extract all text content from image
        """

        start_time = time.time()
        response = self.paddle_predict()

        print("--- %s seconds ---" % (time.time() - start_time))
        return response


class Prediction(object):
    def __init__(self, image_path: str, image_id: int):
        self.image_id = image_id
        self.image_path = image_path

    def get_prediction_bounding_box(self, conf: float = 0.2):
        """
        Get document layout and detect bounding boxes
        """

        res = yolov10.predict(
            source=self.image_path,
            imgsz=1024,
            conf=conf,
            iou=0.3,
            device="cpu",
            max_det=1000,
            agnostic_nms=False,
            verbose=True,
        )

        if not res:
            return []

        bounding_boxes = []

        for box in res[0].boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords
            bounding_boxes.append(
                {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
            )

        return bounding_boxes

    def _predict_bounding_box_content(self, image, bounding_box):
        """
        Detect a single bounding box text content
        """

        if not bounding_box:
            return None

        x1 = bounding_box["x1"]
        y1 = bounding_box["y1"]
        x2 = bounding_box["x2"]
        y2 = bounding_box["y2"]

        crop_image = image[y1:y2, x1:x2]

        try:
            with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp:
                cv2.imwrite(tmp.name, crop_image)
                frame = Frame(image_path=tmp.name)
                content = frame.run_predict()
                if content:
                    return {
                        "image_id": self.image_id,
                        "bounding_box": bounding_box,
                        "content": str(content),
                    }
        except:
            raise Exception("Failed to predict bounding box content")

    def run_predict(self, max_workers=5):
        """
        Extract all bounding boxes text content
        """

        pred_frames = []
        image = cv2.imread(self.image_path)

        bounding_boxes = self.get_prediction_bounding_box()

        if not bounding_boxes:
            return ValueError("No bounding box detected!")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pred_frames = list(
                executor.map(
                    lambda bbox: self._predict_bounding_box_content(image, bbox),
                    bounding_boxes,
                )
            )
        return pred_frames
