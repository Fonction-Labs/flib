import pytest
from flib.models.llms.bedrock import BedRockLLMModel

def test_bedrock_run():
    model = BedRockLLMModel(model_name="mistral.mistral-large-2402-v1:0")
    response = model.run(messages=[{"role": "user", "content": "Hello!"}])
    assert isinstance(response, str)

def test_openai_gpt_model_run_batch():
    model = BedRockLLMModel(model_name="mistral.mistral-large-2402-v1:0")
    responses = model.run_batch(list_messages=[[{"role": "user", "content": "Hello!"}]], parallel=False)
    assert isinstance(responses, list)
    assert all(isinstance(resp, str) for resp in responses)
