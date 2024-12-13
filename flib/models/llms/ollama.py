from typing import Generator
import ollama
from PIL import Image
from tqdm import tqdm
from warnings import warn

from flib.utils.images import encode_image_base64
from .base_llm import BaseLLM, BaseEmbedding

class OllamaModel(BaseLLM):
    """
    A model for interacting with Ollama's chat models.

    Attributes:
        model_name (str): The name of the Ollama model to use.
        context_window_token_size (int): The maximum number of tokens the model can handle in a single request.
    """
    def __init__(self, model_name: str):
        ollama.pull(model_name)
        self.model_name = model_name

    def run(self, messages, temperature: float = 0.0, stream: bool = False, json_output: bool = False) -> str:
        """
        Runs the model with the provided messages and returns the generated response.

        Args:
            messages (list): A list of messages to send to the model.
            temperature (float): Sampling temperature for randomness in responses.

        Returns:
            str: The generated response from the model.
        """
        if temperature != 0:
            warn(
                "Change of temperature is not handled by OllamaModel models (ollama does not allow so). Temperature is still 0."
            )
        if json_output:
            response = ollama.chat(model=self.model_name, messages=messages, stream=stream, format="json")
        else:
            response = ollama.chat(model=self.model_name, messages=messages, stream=stream)
        if stream:
            return parse_stream(response)
        else:
            return response["message"]["content"]

    def run_batch(
        self, list_messages: list[dict], temperature: float = 0.0, stream: bool = False, json_output: bool = False, parallel: bool = False, n_jobs: int = 8
    ) -> list[str]:
        """
        Runs the model in batch mode with the provided list of messages.

        Args:
            list_messages (list): A list of message lists to send to the model.
            temperature (float): Sampling temperature for randomness in responses.
            parallel (bool): Whether to run the requests in parallel.

        Returns:
            list[str]: A list of responses from the model.
        """
        if parallel:
            warn(
                "Parallelization is not available for Ollama models. Batch will not be parallelized."
            )
        return [self.run(messages, temperature=temperature, stream=stream, json_output=json_output) for messages in tqdm(list_messages)]


class OllamaEmbeddingModel(BaseEmbedding):
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


def parse_stream(stream):
    for chunk in stream:
        if chunk["message"]["content"]
            yield chunk["content"]
        else:
            return "\n \n"
