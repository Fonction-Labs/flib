from PIL import Image

from flib.utils.images import encode_image_base64

# TODO: parallelization?


class OllamaModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        ollama.pull(self.model_name)

    def run(self, prompt: str, images: list[Image]) -> str:
        images = [encode_image_base64(image) for image in images]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
                + [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    }
                    for image in images
                ],
            },
        ]
        return ollama.chat(model=self.model_name, messages=messages)["message"][
            "content"
        ]


class OllamaEmbeddingModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        ollama.pull(self.model_name)

    def run(self, prompt: str) -> list[float]:
        return ollama.embed(model=self.model_name, prompt=prompt)["embedding"]
