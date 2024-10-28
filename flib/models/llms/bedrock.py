import boto3
from botocore.config import Config
import json
from typing import Generator
from botocore.exceptions import ClientError
from flib.utils.parallel import ParallelTqdm
from joblib import delayed
from tqdm import tqdm
import itertools
from ..base import BaseModel

class BedRockLLMModel(BaseModel):
    """
    A model for interacting with Amazon Bedrock's LLMs.

    Attributes:
        model_name (str): The name of the Bedrock model to use.
        client: The Bedrock client for making API calls.
        context_window_token_size (int): The maximum number of tokens the model can handle in a single request.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = get_bedrock_client()
        self.context_window_token_size = 1024 # TODO get the context window

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
            bedrock=self.client,
            temperature=temperature,
            json_output=json_output
        )

    def run_batch(
        self,
        list_messages,
        temperature: float = 0.0,
        parallel: bool = False,
        n_jobs: int = 8
    ):
        """
        Runs the model in batch mode with the provided list of messages.

        Args:
            list_messages (list): A list of message dictionaries to send to the model.
            temperature (float): Sampling temperature for randomness in responses.
            parallel (bool): Whether to run the requests in parallel.
            n_jobs (int): Number of jobs to run in parallel.

        Returns:
            list: A list of responses from the model.
        """
        if parallel:
            return ParallelTqdm(n_jobs=n_jobs, prefer="threads", total_tasks=len(list_messages))(
                delayed(self.run)(messages, temperature)
                for messages in list_messages
            )
        return [
            self.run(messages, temperature)
            for messages in tqdm(list_messages)
        ]

def get_bedrock_client():
    config = Config(read_timeout=1000)
    return boto3.client(service_name="bedrock-runtime", config=config)

def get_embeddings_bedrock(prompt: str, model_id: str, bedrock):
    json_request = {"inputText": prompt}
    body = json.dumps(json_request)

    try:
        response = bedrock.invoke_model(body=body, modelId=model_id)
        response_body = response.get('body').read()
        embedding = json.loads(response_body)['embedding']
        return embedding
    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)


def get_llm_answer_bedrock(messages: str, model_id: str, bedrock, temperature: float = 0.0, json_output: bool = False, stream: bool = False) -> str:
    native_request = {
        'messages': messages
    }
    if json_output:
        raise Warning("Json output not available for Bedrock Models")
        pass

    request = json.dumps(native_request)

    if not stream:
        try:
            response = bedrock.invoke_model(modelId=model_id, body=request)
        except (ClientError, Exception) as e:
            print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
            exit(1)

        model_response = json.loads(response["body"].read())
        return model_response["choices"][0]["message"]["content"]
    
    else:
        try:
            streaming_response = bedrock.invoke_model_with_response_stream(
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
