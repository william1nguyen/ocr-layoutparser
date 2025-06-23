import base64


def image_paths_to_base64(image_paths: list[str]):
    base64_images = []
    for image_path in image_paths:
        with open(image_path, 'rb') as f:
            data = f.read()
            base64_string = base64.b64encode(data).decode("utf-8")
            base64_images.append(base64_string)

    return base64_images