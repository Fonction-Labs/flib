from typing import Optional
from openai import OpenAI
from joblib import delayed
from PIL import Image
from tqdm import tqdm

from flib.utils.images import encode_image_base64
from flib.utils.parallel import ParallelTqdm
from .base import BaseModel

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
        self, prompt: str, images: Optional[list[Image.Image]] = None, temperature: float = 0.0
    ) -> Optional[str]:
        if images is None:
            images = []
        encoded_images = [encode_image_base64(image) for image in images]

        from openai.types.chat_completion_system_message_param import ChatCompletionSystemMessageParam
        from openai.types.chat_completion_user_message_param import ChatCompletionUserMessageParam
        system_message = ChatCompletionSystemMessageParam(content="You are a helpful assistant.")
        user_message = ChatCompletionUserMessageParam(content=[{"type": "text", "text": prompt}] + [{ "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"},} for image in encoded_images])
        messages = (system_message, user_message)
        #messages = [
        #    {"role": "system", "content": "You are a helpful assistant."},
        #    {
        #        "role": "user",
        #        "content": [{"type": "text", "text": prompt}]
        #        + [
        #            {
        #                "type": "image_url",
        #                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
        #            }
        #            for image in images
        #        ],
        #    },
        #]
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
        prompts: list[str],
        list_images: (None | list[(None | list[Image.Image])]) = None,
        temperature: float = 0.0,
        parallel: bool = False,
    ):
        if list_images is None:
            list_images = [None] * len(prompts)
        if parallel:
            return ParallelTqdm(n_jobs=8, prefer="threads", total_tasks=len(prompts))(
                delayed(self.run)(prompt, images, temperature)
                for prompt, images in zip(prompts, list_images)
            )
        return [
            self.run(prompt, images, temperature)
            for prompt, images in tqdm(zip(prompts, list_images))
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
