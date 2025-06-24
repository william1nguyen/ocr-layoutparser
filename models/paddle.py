from paddleocr import PaddleOCR

paddle = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)


def predict(image_path: str):
    result = paddle.predict(input=image_path)
    contents = []
    for res in result:
        content = " ".join(res["rec_texts"])
        contents.append(content)
    return " ".join(contents)
