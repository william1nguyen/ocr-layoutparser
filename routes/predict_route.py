import os
import json
import tempfile
from fastapi import APIRouter, File, UploadFile
from services.predict_service import *

predict_router = APIRouter(prefix="/predict", tags=["predict"])


@predict_router.post("/content")
async def detect_content(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp:
            tmp.write(contents)
            tmp.flush()
            frame = Frame(image_path=tmp.name)
            content = frame.run_predict()
            return content
    except Exception as err:
        raise err


@predict_router.post("")
async def detect_bounding_box_content(files: list[UploadFile] = File(...)):
    bounding_box_contents = []
    for id in range(len(files)):
        file = files[id]

        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp:
            tmp.write(contents)
            tmp.flush()

            prediction = Prediction(image_path=tmp.name, image_id=id)
            bounding_box_content = prediction.run_predict()
            bounding_box_contents.append(bounding_box_content)

    return contents
