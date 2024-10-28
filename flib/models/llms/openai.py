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
    """
    A model for interacting with OpenAI's GPT models.

    Attributes:
        model_name (str): The name of the OpenAI model to use.
        client (OpenAI): The OpenAI client for making API calls.
        context_window_token_size (int): The maximum number of tokens the model can handle in a single request.
    """

    def __init__(self, model_name: str, api_key: str):
        """
        Initializes the OpenAIGPTModel with the specified model name and API key.

        Args:
            model_name (str): The name of the OpenAI model.
            api_key (str): The API key for authenticating with OpenAI.
        """
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        self.context_window_token_size = MODEL_NAME_TO_CONTEXT_WINDOW_TOKEN_SIZE[model_name]

    def run(self, messages, temperature: float = 0.0) -> Optional[str]:
        """
        Runs the model with the provided messages and returns the generated response.

        Args:
            messages (list): A list of messages to send to the model.
            temperature (float): Sampling temperature for randomness in responses.

        Returns:
            Optional[str]: The generated response from the model.
        """
        return (
            self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
            )
            .choices[0]
            .message.content
        )

    def run_batch(self, list_messages, temperature: float = 0.0, parallel: bool = False, n_jobs: int = 8):
        """
        Runs the model in batch mode with the provided list of messages.

        Args:
            list_messages (list): A list of message lists to send to the model.
            temperature (float): Sampling temperature for randomness in responses.
            parallel (bool): Whether to run the requests in parallel.
            n_jobs (int): Number of jobs to run in parallel.

        Returns:
            list: A list of responses from the model.
        """
        if parallel:
            return ParallelTqdm(n_jobs=n_jobs, prefer="threads", total_tasks=len(list_messages))(
                delayed(self.run)(message, temperature) for message in list_messages
            )
        return [self.run(message, temperature) for message in list_messages]

class OpenAIEmbeddingModel(BaseModel):
    """
    A model for generating embeddings using OpenAI's embedding models.

    Attributes:
        model_name (str): The name of the OpenAI embedding model to use.
        client (OpenAI): The OpenAI client for making API calls.
        embedding_vector_size (int): The size of the embedding vector produced by the model.
    """

    def __init__(self, model_name: str, api_key: str):
        """
        Initializes the OpenAIEmbeddingModel with the specified model name and API key.

        Args:
            model_name (str): The name of the OpenAI embedding model.
            api_key (str): The API key for authenticating with OpenAI.
        """
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        self.embedding_vector_size = MODEL_NAME_TO_EMBEDDING_VECTOR_SIZE[model_name]

    def run(self, prompt: str) -> list[float]:
        """
        Generates an embedding for the provided prompt.

        Args:
            prompt (str): The input text for which to generate an embedding.

        Returns:
            list[float]: The generated embedding vector.
        """
        return (
            self.client.embeddings.create(input=[prompt], model=self.model_name)
            .data[0]
            .embedding
        )

    def run_batch(self, prompts: list[str], parallel: bool = False) -> list[list[float]]:
        """
        Generates embeddings for a batch of prompts.

        Args:
            prompts (list[str]): A list of prompts to generate embeddings for.
            parallel (bool): Whether to run the requests in parallel.

        Returns:
            list[list[float]]: A list of generated embedding vectors.
        """
        if parallel:
            return ParallelTqdm(n_jobs=8, prefer="threads", total_tasks=len(prompts))(
                delayed(self.run)(prompt) for prompt in prompts
            )
        return [self.run(prompt) for prompt in tqdm(prompts)]