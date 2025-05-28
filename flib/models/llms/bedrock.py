import boto3
from botocore.config import Config
import json
from typing import Generator, Optional, Type
from pydantic import BaseModel
from warnings import warn
from botocore.exceptions import ClientError
from flib.utils.parallel import ParallelTqdm
from joblib import delayed
from tqdm import tqdm
import itertools
from .base_llm import BaseLLM
from .utils import clean_json_output

class BedRockLLMModel(BaseLLM):
    """
    A model for interacting with Amazon Bedrock's LLMs.

    Attributes:
        model_name (str): The name of the Bedrock model to use.
        client: The Bedrock client for making API calls.
    """
    def __init__(self, model_name: str, max_tokens: int):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.client = get_bedrock_client()

    def run(
        self, messages: list[dict[str, str]], temperature: float = 1.0, top_p: float = None, top_k: int = None, stop_sequences: list[str] = None, stream: bool = False, json_output: bool = False, text_format: Optional[Type[BaseModel]] = None
    ) -> (Generator[str, str, None] | str):
        """
        Runs the model with the provided messages and returns the generated response.

        Args:
            messages (dict): A dictionary of messages to send to the model.
            temperature (float): Sampling temperature for randomness in responses.
            stream (bool): Whether to stream the response.
            json_output (bool): Whether to return the response in JSON format.

        Returns:
            (Generator[str, str, None] | str): The generated response from the model, either as a string or a generator.
        """
        if text_format is not None:
            raise ValueError("Text format is not yet supported for Bedrock wrapper")
        return get_llm_answer_bedrock(
            messages=messages,
            model_id=self.model_name,
            client=self.client,
            max_tokens=self.max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
            json_output=json_output,
            stream=stream,
        )

def get_bedrock_client():
    config = Config(read_timeout=1000)
    return boto3.client(service_name="bedrock-runtime", config=config)

def get_embeddings_bedrock(prompts: list[str], model_id: str, client, input_type: str = "search_document"):
    # input_type can be [search_document, search_query, classification, clustering]

    # json_request = {"inputText": prompt}
    json_request = {"texts": prompts, "input_type": input_type, "truncate": "END"}
    body = json.dumps(json_request)

    try:
        response = client.invoke_model(body=body,
                                        modelId=model_id,
                                        accept='application/json',
                                        contentType='application/json')
        response_body = response.get('body').read()
        embeddings = json.loads(response_body)['embeddings']
        return embeddings

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)


def get_llm_answer_bedrock(messages: list[dict[str, str]], model_id: str, client, max_tokens: int, temperature: float = 1.0, top_p: float = None, top_k: int = None, stop_sequences: list[str] = None, json_output: bool = False, stream: bool = False) -> str:

    system_messages = [m for m in messages if m["role"] == "system"]
    messages = [m for m in messages if m["role"] != "system"]

    native_request = { "messages": messages,
                       "max_tokens": max_tokens,
                       "anthropic_version":
                       "bedrock-2023-05-31",
                       "temperature": temperature }

    if len(system_messages) > 0:
        native_request["system"] = system_messages[0]["content"]

    if top_p is not None:
        native_request["top_p"] = top_p

    if top_k is not None:
        native_request["top_k"] = top_k

    if stop_sequences is not None:
        native_request["stop_sequences"] = stop_sequences

    if json_output:
        warn("Json output not available for Bedrock Models")

    request = json.dumps(native_request)

    if not stream:
        try:
            response = client.invoke_model(modelId=model_id, body=request)
        except (ClientError, Exception) as e:
            print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
            exit(1)

        model_response = json.loads(response["body"].read())

        if json_output:
            return clean_json_output(model_response["content"][0]["text"])

        return model_response["content"][0]["text"]

    else:
        try:
            streaming_response = client.invoke_model_with_response_stream(
                modelId=model_id, body=request
            )

        except (ClientError, Exception) as e:
            print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
            exit(1)

        return parse_stream(streaming_response["body"])


def parse_stream(stream):
    for event in stream:
        chunk = event.get('chunk')
        message = json.loads(chunk.get("bytes").decode())
        chunk = json.loads(event["chunk"]["bytes"])
        chunk = chunk["choices"][0]
        yield chunk["message"].get("content")

        if chunk.get("stop_reason"):
            return "\n \n"

from typing import Optional, Type, Dict, Any
from pydantic import BaseModel

def convert_pydantic_to_bedrock_tool(
    model: Type[BaseModel],
    description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Converts a Pydantic model to a tool description for the Amazon Bedrock Converse API.

    Args:
        model: The Pydantic model class to convert
        description: Optional description of the tool's purpose

    Returns:
        Dict containing the Bedrock tool specification

    source: https://freedium.cfd/https://medium.com/@dminhk/structured-output-with-amazon-bedrock-converse-api-4e85d1f602c4
    """
    # Validate input model
    if not isinstance(model, type) or not issubclass(model, BaseModel):
        raise ValueError("Input must be a Pydantic model class")

    name = model.__name__
    input_schema = model.model_json_schema()
    tool = {
        'toolSpec': {
            'name': name,
            'description': description or f"{name} Tool",
            'inputSchema': {'json': input_schema }
        }
    }
    return tool

