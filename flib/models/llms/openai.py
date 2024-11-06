from typing import Optional, Generator
from openai import OpenAI
from joblib import delayed
from PIL import Image
from tqdm import tqdm

from flib.utils.images import encode_image_base64
from flib.utils.parallel import ParallelTqdm
from .base_llm import BaseLLM, BaseEmbedding

class OpenAIGPTModel(BaseLLM):
    """
    A model for interacting with OpenAI's GPT models.

    Attributes:
        model_name (str): The name of the OpenAI model to use.
        client (OpenAI): The OpenAI client for making API calls.
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

    def run(
        self, messages, temperature: float = 0.0, stream: bool = False, json_output: bool = False
    ) -> (Generator[str, str, None] | str):
        """
        Runs the model with the provided messages and returns the generated response.

        Args:
            messages (list): A list of messages to send to the model.
            temperature (float): Sampling temperature for randomness in responses.
            stream (bool): Whether to stream the response.
            json_output (bool): Whether to return the response in JSON format.

        Returns:
            (Generator[str, str, None] | str): The generated response from the model, either as a string or a generator.
        """

        if json_output:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=stream,
                response_format={ "type": "json_object" },
            )

        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=stream,
            )

        if not stream:
            return response.choices[0].message.content
        else:
            return parse_stream(response)

class OpenAIEmbeddingModel(BaseEmbedding):
    """
    A model for generating embeddings using OpenAI's embedding models.

    Attributes:
        model_name (str): The name of the OpenAI embedding model to use.
        client (OpenAI): The OpenAI client for making API calls.
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

def parse_stream(stream):
    for chunk in stream:
        if len(chunk.choices) > 0:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
            else:
                return "\n \n"
