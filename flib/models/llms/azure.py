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
from config import AZURE_OPENAI_ENDPOINT, AZURE_INFERENCE_ENDPOINT
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from ..base import BaseModel

class AzureOpenaiModel(BaseModel):
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.client = get_azure_client()
        self.context_window_token_size = 1024

    def run(
        self, messages, temperature: float = 0.0, stream: bool = False, json_output: bool = False
    ) -> (Generator[str, str, None] | str):


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


def get_message_azure(message):
    match message["role"]:
        case "system":
            return SystemMessage(content=messages["content"])
        case "user":
            return UserMessage(content=messages["content"])
        case "assistant":
            return AssistantMessage(content=messages["content"])



class AzureInferenceModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = get_azure_completion_client()
        self.context_window_token_size = 1024

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


def get_azure_completion_client():
    client = ChatCompletionsClient(
        endpoint=AZURE_INFERENCE_ENDPOINT,
        credential=AzureKeyCredential(os.environ["AZURE_INFERENCE_CREDENTIAL"]),
    )
    return client

def get_azure_client():
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version='2024-06-01',
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )
    return client

def parse_stream(stream):
    for chunk in stream:
        if len(chunk.choices) > 0:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
            else:
                return "\n \n"
