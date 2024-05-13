from openai import OpenAI
from PIL import Image

from flib.utils.images import encode_image_base64

class OpenAIGPTModel():
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def run(self, prompt: str, images: list[Image]):
        images = [encode_image_base64(image) for image in images]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}} for image in images],}]

        return self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.,
            ).choices[0].message.content
