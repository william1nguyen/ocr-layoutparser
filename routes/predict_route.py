import tempfile
from typing import List
from fastapi import APIRouter, File, Query, UploadFile
from services.predict_service import *

predict_router = APIRouter(prefix="/predict", tags=["predict"])


@predict_router.post('/content')
async def detect_content(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=True, suffix='.png') as tmp:
            tmp.write(contents)
            tmp.flush()
            frame = Frame(image_path=tmp.name)
            content = frame.run_predict()
            return content
    except Exception as err:
        raise err