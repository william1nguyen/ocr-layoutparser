from env import env
from . import vlm

gemini = vlm.OpenAI(
    base_url=env.GEMINI_BASE_URL,
    api_key=env.GEMINI_API_KEY
)

gemini_model = 'gemini-2.5-flash-preview-05-20'

def prompting(message: str, image_paths: list[str]):
    return vlm.prompting(
        client=gemini,
        model=gemini_model,
        message=message,
        image_paths=image_paths
    )