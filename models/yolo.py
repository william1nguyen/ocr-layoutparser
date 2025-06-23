from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download

pretrained_weight = hf_hub_download(
    repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
    filename="doclayout_yolo_docstructbench_imgsz1024.pt",
)
yolov10 = YOLOv10(pretrained_weight)
