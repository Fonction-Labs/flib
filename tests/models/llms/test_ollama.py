# import pytest
# from flib.models.llms.ollama import OllamaModel

# def test_ollama_model_run():
#     model = OllamaModel(model_name="mistral")
#     response = model.run(messages=[{"role": "user", "content": "Hello!"}])
#     assert isinstance(response, str)
#     assert len(response) > 0  # Ensure that the response is not empty
