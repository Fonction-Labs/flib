import pytest
from flib.models.llms.openai import OpenAIGPTModel
from flib.models.llms.openai import OpenAIEmbeddingModel
import os

def test_openai_gpt_model_run():
    model = OpenAIGPTModel(model_name="gpt-3.5-turbo", api_key=os.environ["OPENAI_API_KEY"])
    response = model.run(messages=[{"role": "user", "content": "Hello!"}])
    assert isinstance(response, str)

def test_openai_gpt_model_run_batch():
    model = OpenAIGPTModel(model_name="gpt-3.5-turbo", api_key=os.environ["OPENAI_API_KEY"])
    responses = model.run_batch(list_messages=[[{"role": "user", "content": "Hello!"}]], parallel=False)
    assert isinstance(responses, list)
    assert all(isinstance(resp, str) for resp in responses)

def test_openai_embedding_model_run():
    model = OpenAIEmbeddingModel(model_name="text-embedding-3-small", api_key=os.environ["OPENAI_API_KEY"])
    embedding = model.run(prompt="Hello!")
    assert isinstance(embedding, list)
    assert len(embedding) == 1536
