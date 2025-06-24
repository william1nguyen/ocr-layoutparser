import json
from openai import OpenAI
from utils import image


def prompting(
    client: OpenAI, model: str, message: str, image_paths: list[str], retries=1
):
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

    messages = [{"role": "user", "content": content}]

    for _ in range(retries):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content.strip()

            if content.startswith("```json"):
                content = content.removeprefix("```json").strip()
            if content.endswith("```"):
                content = content.removesuffix("```").strip()

            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON returned: {e}") from e
        except:
            pass

    return {"error": "No valid JSON content returned from response."}
