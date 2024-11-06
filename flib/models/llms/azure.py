import os
import json
from typing import Generator
from botocore.exceptions import ClientError
from flib.utils.parallel import ParallelTqdm
from joblib import delayed
from tqdm import tqdm
import itertools
from openai import AzureOpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import ChatCompletionsResponseFormatJSON
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from .base_llm import BaseLLM
from .openai import OpenAIGPTModel


class AzureOpenaiModel(OpenAIGPTModel):
    """
    A model for interacting with Azure OpenAI's chat completions.

    Attributes:
        model_name (str): The name of the Azure OpenAI model to use.
        client (AzureOpenAI): The Azure OpenAI client for making API calls.
    """
    def __init__(self, endpoint: str, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.client = get_azure_client(endpoint)


class AzureInferenceModel(BaseLLM):
    def __init__(self, endpoint: str, model_name: str):
        self.model_name = model_name
        self.client = get_azure_completion_client(endpoint)

    def run(
        self, messages, temperature: float = 0.0, stream: bool = False, json_output: bool = False
    ) -> (Generator[str, str, None] | str):

        if json_output:
            response = self.client.complete(
                messages=list(map(get_message_azure, messages)),
                temperature=temperature,
                stream=stream,
                response_format=ChatCompletionsResponseFormatJSON()
            )

        else:
            response = self.client.complete(
                messages=list(map(get_message_azure, messages)),
                temperature=temperature,
                stream=stream,
            )

        if not stream:
            return response.choices[0].message.content
        else:
            return parse_stream(response)

def get_azure_client(endpoint):
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version='2024-06-01',
        azure_endpoint=endpoint
    )
    return client

def get_azure_completion_client(endpoint):
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(os.environ["AZURE_INFERENCE_CREDENTIAL"]),
    )
    return client

def get_message_azure(message):
    match message["role"]:
        case "system":
            return SystemMessage(content=messages["content"])
        case "user":
            return UserMessage(content=messages["content"])
        case "assistant":
            return AssistantMessage(content=messages["content"])
