from google import genai
from google.genai import types
from anthropic import AnthropicVertex, Anthropic

from typing import Generator, Optional, Type, Dict, Any, List
from pydantic import BaseModel
import json
import os
from tqdm import tqdm
from warnings import warn
from .base_llm import BaseLLM
from .utils import clean_json_output

class AnthropicVertexLLMModel(BaseLLM):

    def __init__(self, model_name: str, project_id:str, location: str):
        self.model_name = model_name
        # self.client = AnthropicVertex(region=location, project_id=project_id)
        self.client = Anthropic()

    def run(
        self, messages: list[dict[str, str]], max_tokens: int, temperature: float = 1.0, top_p: float = None, top_k: int = None, stop_sequences: list[str] = None, stream: bool = False, json_output: bool = False, text_format: Optional[Type[BaseModel]] = None
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
        return get_llm_answer_anthropic(
            messages=messages,
            model_id=self.model_name,
            client=self.client,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
            json_output=json_output,
            stream=stream,
        )



def get_llm_answer_anthropic(messages: list[dict[str, str]], model_id: str, client, max_tokens: int, temperature: float = 1.0, top_p: float = None, top_k: int = None, stop_sequences: list[str] = None, json_output: bool = False, stream: bool = False) -> str:

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
            message = client.messages.create(
                system=system_messages[0]["content"],
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                stop_sequences=stop_sequences,
                messages=messages,
                model="claude-3-5-sonnet-latest",
            )

        except Exception as e:
            print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
            exit(1)

        if json_output:
            return clean_json_output(message.content[0].text)

        return message.content[0].text

    else:
        # TODO
        pass

import os
from typing import List, Optional
import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

vertexai.init(project=os.getenv("GOOGLE_PROJECT_ID"))

def get_embeddings_vertex(prompts: list[str], model_id: str, task: str = "RETRIEVAL_DOCUMENT"): #, dimensionality: Optional[int] = 256,):
    
    # https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api?hl=fr

    try:
        model = TextEmbeddingModel.from_pretrained(model_id)
        inputs = [TextEmbeddingInput(prompt, task) for prompt in prompts]
        #kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}
        kwargs = {}
        embeddings = model.get_embeddings(inputs, **kwargs)
        return [embedding.values for embedding in embeddings]

    except Exception as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")


class VertexAILLMModel(BaseLLM):
    """
    A model for interacting with Google Cloud's Vertex AI LLMs.
    Attributes:
        model_name (str): The name of the Vertex AI model to use.
        client: The Vertex AI client for making API calls.
    """
    def __init__(self, model_name: str):
        self.client = genai.Client() # Will automatically fetch GOOGLE_API_KEY env variable

    def run(
    self, messages: list[dict[str, str]], temperature: float = 1.0, top_p: float = None, top_k: int = None, 
    max_tokens: int = None,
    stop_sequences: list[str] = None, stream: bool = False, json_output: bool = False, 
    text_format: Optional[Type[BaseModel]] = None
) -> (Generator[str, str, None] | str):
        """
        Runs the model with the provided messages and returns the generated response.
        """
        # Convert messages to the format expected by the genai library
        contents = []
        system_instruction = None
        
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
                continue
                
            role = "model" if msg["role"] == "assistant" else msg["role"]
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })
            
        # Generation config
        generation_config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction
        )
        
        if top_p is not None:
            generation_config.top_p = top_p
        
        if top_k is not None:
            generation_config.top_k = top_k

        if max_tokens is not None:
            generation_config.max_output_tokens = max_tokens
            
        if stop_sequences:
            generation_config.stop_sequences = stop_sequences
            
        if json_output and text_format is not None:
            # Set response format for structured JSON
            schema = text_format.model_json_schema()
            generation_config.response_schema = schema
        elif json_output:
            generation_config.response_mime_type = "application/json"
            
        try:
            # Use the updated client.models.generate_content syntax
            if not stream:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=generation_config,
                )
                
                if json_output:
                    return clean_json_output(response.text)
                return response.text
            else:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=generation_config,
                    stream=True
                )
                return self._stream_response_generator(response)
                
        except Exception as e:
            print(f"ERROR: Can't invoke '{self.model_name}'. Reason: {e}")
            exit(1)
            
    def _stream_response_generator(self, response_stream):
        """Parse streaming responses from Gemini API."""
        for chunk in response_stream:
            if hasattr(chunk, 'text'):
                yield chunk.text
            elif hasattr(chunk, 'parts') and chunk.parts:
                for part in chunk.parts:
                    if hasattr(part, 'text') and part.text:
                        yield part.text
        yield "\n \n"
