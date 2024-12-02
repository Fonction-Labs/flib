# flib

flib is a library for AI models and utilities created by and for Fonction Labs. It provides a collection of models for various AI tasks, including natural language processing and image segmentation.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Models](#models)
  - [Utilities](#utilities)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Features

- Support for multiple AI models including OpenAI, Ollama, Bedrock, and more.
- Utilities for image processing and text chunking.
- Parallel processing capabilities for efficient model execution.

## Installation

To install the project, you can use Poetry. Make sure you have Poetry installed, then run:

```bash
poetry install
```

This will install all the required dependencies specified in the `pyproject.toml` file.

## Usage

### Models

The library includes several models for different tasks. Here are some examples of how to use them:

#### OpenAI GPT Model

```python
from flib.models.llms.openai import OpenAIGPTModel
import os

model = OpenAIGPTModel(model_name="gpt-3.5-turbo", api_key=os.environ["OPENAI_API_KEY"])
response = model.run(messages=[{"role": "user", "content": "Hello!"}])
print(response)
```

#### Ollama Model

```python
from flib.models.llms.ollama import OllamaModel

model = OllamaModel(model_name="mistral")
response = model.run(messages=[{"role": "user", "content": "Hello!"}])
print(response)
```

#### Bedrock Model

```python
from flib.models.llms.bedrock import BedRockLLMModel

model = BedRockLLMModel(model_name="mistral.mistral-large-2402-v1:0")
response = model.run(messages=[{"role": "user", "content": "Hello!"}])
print(response)
```

### Utilities

The library also provides utility functions for image processing and text chunking.

#### Image Processing

```python
from flib.utils.images import load_image, encode_image_base64

image = load_image("path/to/image.png")
encoded_image = encode_image_base64(image)
```

#### Text Chunking

```python
from flib.utils.chunk_text import get_text_chunks

text = "This is a test text for chunking."
chunks = get_text_chunks(text, chunk_size=10, chunk_overlap=2)
print(chunks)
```

## Testing

To run the tests, you can use pytest. Make sure you have pytest installed, then run:

```bash
pytest
```

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
