import ollama
from PIL import Image
from tqdm import tqdm
from warnings import warn

from flib.utils.images import encode_image_base64
from .base import BaseModel

class OllamaModel(BaseModel):
    def __init__(self, model_name: str):
        ollama.pull(model_name)
        self.model_name = model_name

    def run(self, prompt: str, temperature: float = 0.0) -> str:
        if temperature != 0:
            warn(
                "Change of temperature is not handled by OllamaModel models (ollama does not allow so). Temperature is still 0."
            )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        return ollama.chat(model=self.model_name, messages=messages)["message"][
            "content"
        ]

    def run_batch(
        self, prompts: list[str], temperature: float = 0.0, parallel: bool = False
    ) -> list[str]:
        if parallel:
            warn(
                "Parallelization is not available for Ollama models. Batch will not be parallelized."
            )
        return [self.run(prompt, temperature) for prompt in tqdm(prompts)]


class OllamaEmbeddingModel(BaseModel):
    def __init__(self, model_name: str):
        ollama.pull(model_name)
        self.model_name = model_name

    def run(self, prompt: str) -> list[float]:
        return ollama.embeddings(model=self.model_name, prompt=prompt)["embedding"]

    def run_batch(self, prompts: list[str], parallel: bool = False) -> list[list[float]]:
        if parallel:
            warn(
                "Parallelization is not available for Ollama models. Batch will not be parallelized."
            )
        return [self.run(prompt) for prompt in tqdm(prompts)]
