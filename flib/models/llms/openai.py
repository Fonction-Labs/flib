from typing import Optional
from openai import OpenAI
from joblib import delayed
from PIL import Image
from tqdm import tqdm

from flib.utils.images import encode_image_base64
from flib.utils.parallel import ParallelTqdm
from ..base import BaseModel

MODEL_NAME_TO_CONTEXT_WINDOW_TOKEN_SIZE = {
    "gpt-3.5-turbo": 4096,
    "gpt-4-turbo": 128000,
}

MODEL_NAME_TO_EMBEDDING_VECTOR_SIZE = {
    "text-embedding-3-small": 1536,
}


class OpenAIGPTModel(BaseModel):
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        self.context_window_token_size = MODEL_NAME_TO_CONTEXT_WINDOW_TOKEN_SIZE[
            model_name
        ]

    def run(
        self, messages, temperature: float = 0.0
    ) -> Optional[str]:
        return (
            self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
            )
            .choices[0]
            .message.content
        )

    def run_batch(
        self,
        list_messages,
        temperature: float = 0.0,
        parallel: bool = False,
        n_jobs: int = 8
    ):
        if parallel:
            return ParallelTqdm(n_jobs=8, prefer="threads", total_tasks=len(list_messages))(
                delayed(self.run)(message, temperature)
                for message in list_messages
            )
        return [
            self.run(message, temperature)
            for message in list_messages
        ]


class OpenAIEmbeddingModel(BaseModel):
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        self.embedding_vector_size = MODEL_NAME_TO_EMBEDDING_VECTOR_SIZE[model_name]

    def run(self, prompt: str) -> list[float]:
        return (
            self.client.embeddings.create(input=[prompt], model=self.model_name)
            .data[0]
            .embedding
        )

    def run_batch(
        self,
        prompts: list[str],
        parallel: bool = False,
    ) -> list[list[float]]:
        if parallel:
            return ParallelTqdm(n_jobs=8, prefer="threads", total_tasks=len(prompts))(
                delayed(self.run)(prompt) for prompt in prompts
            )
        return [self.run(prompt) for prompt in tqdm(prompts)]
