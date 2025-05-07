import boto3
from botocore.config import Config
import json
from typing import Generator
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
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = get_bedrock_client()

    def run(
        self, messages: dict, temperature: float = 0.0, stream: bool = False, json_output: bool = False
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
        return get_llm_answer_bedrock(
            messages=messages,
            model_id=self.model_name,
            client=self.client,
            temperature=temperature,
            json_output=json_output,
            stream=stream
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


# Default temp is 1 on Bedrock ?
def get_llm_answer_bedrock(messages: str, model_id: str, client, temperature: float = 1.0, json_output: bool = False, stream: bool = False) -> str:
    
    system_messages = [m for m in messages if m["role"] == "system"]
    messages = [m for m in messages if m["role"] != "system"]

    native_request = { "messages": messages, 
                       "max_tokens": 1000, 
                       "anthropic_version":
                       "bedrock-2023-05-31", 
                       "temperature": temperature } 

    if len(system_messages) > 0:
        native_request["system"] = system_messages[0]

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
