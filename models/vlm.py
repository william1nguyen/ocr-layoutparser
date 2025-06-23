import json
from openai import OpenAI
from utils import image


def prompting(client: OpenAI, model: str, message: str, image_paths: list[str]):
    content = []

    if message:
        content.append(
            {
                "type": "text",
                "text": message,
            },
        )

    if image_paths:
        base64_images = image.image_paths_to_base64(image_paths=image_paths)
        for base64_image in base64_images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
            )

    message = [{"role": "user", "content": content}]

    response = client.chat.completions.create(
        model=model, messages=message, response_format={"type": "json_object"}
    )

    json_content = response.choices[0].message.content
    try:
        return json.loads(json_content)
    except Exception as err:
        raise err
