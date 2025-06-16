from datetime import datetime
import os
import tempfile
import cv2
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from models.yolo import yolov10

layout_router = APIRouter(prefix="/layout", tags=["layout"])

@layout_router.post('/detect')
async def detect_layout(
    file: UploadFile = File(..., description="Image file to analyze"),
    confidence: float = Query(0.15, ge=0.0, le=1.0, description="Confidence threshold"),
    image_size: int = Query(512, ge=32, le=2048, description="Image size for prediction"),
    device: str = Query("cpu", description="Device to use (cpu or cuda:0)"),
    return_annotated: bool = Query(False, description="Return annotated image as base64")
):
    """
    Detect document layout elements in uploaded image
    Returns JSON with bounding boxes and detection information
    """
    
    if not file.filename:
        raise HTTPException(status_code=404, detail="File not found")

    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    temp_file = None
    try:
        contents = await file.read()

        file_extension = os.path.splitext(file.filename)[1]
        if not file_extension:
            raise HTTPException(status_code=400, detail="Invalid file")
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        temp_file.write(contents)
        temp_file.close()
        
        det_res = yolov10.predict(
            temp_file.name,
            imgsz=image_size,
            conf=confidence,
            device=device
        )

        annotated_frame = det_res[0].plot(pil=True, line_width=5, font_size=20)       

        results = []
        if det_res and len(det_res) > 0:
            boxes = det_res[0].boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    coords = box.xyxy[0].cpu().numpy().tolist()
                    
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = yolov10.names[class_id] if hasattr(yolov10, 'names') else f"class_{class_id}"
                    conf_score = float(box.conf[0].cpu().numpy())
                    
                    bbox_info = {
                        "id": i,
                        "class_name": class_name,
                        "class_id": class_id,
                        "confidence": conf_score,
                        "bbox": {
                            "x1": coords[0],
                            "y1": coords[1],
                            "x2": coords[2],
                            "y2": coords[3]
                        },
                        "width": coords[2] - coords[0],
                        "height": coords[3] - coords[1],
                        "center": {
                            "x": (coords[0] + coords[2]) / 2,
                            "y": (coords[1] + coords[3]) / 2
                        }
                    }
                    results.append(bbox_info)

        response_data = {
            "success": True,
            "filename": file.filename,
            "total_detections": len(results),
            "detections": results,
            "file_info": {
                "filename": file.filename,
                "content_type": file.content_type,
                "file_size": len(contents)
            },
            "parameters": {
                "confidence": confidence,
                "image_size": image_size,
                "device": device
            },
            "processed_at": datetime.now().isoformat()
        }
        
        if return_annotated and det_res and len(det_res) > 0:
            try:
                annotated_frame = det_res[0].plot(pil=True, line_width=5, font_size=20)
                
                import io
                import base64
                
                buffer = io.BytesIO()
                annotated_frame.save(buffer, format='PNG')
                buffer.seek(0)
                annotated_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                response_data["annotated_image"] = f"data:image/jpeg;base64,{annotated_base64}"
                
            except Exception as e:
                response_data["annotation_error"] = f"Failed to create annotated image: {str(e)}"
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    finally:
        if temp_file:
            try:
                os.unlink(temp_file.name)
            except Exception:
                pass