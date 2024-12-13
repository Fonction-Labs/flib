import abc
from typing import Generator

from ..base import BaseModel

class BaseLLM(BaseModel):

    @abc.abstractmethod
    def run(
        self, messages, temperature: float = 0.0, stream: bool = False, json_output: bool = False
    ) -> (Generator[str, str, None] | str):
        pass

    def run_batch(self, list_messages, temperature: float = 0.0, stream: bool = False, json_output: bool = False, parallel: bool = False, n_jobs: int = 8):
        """
        Runs the model in batch mode with the provided list of messages.

        Args:
            list_messages (list): A list of message lists to send to the model.
            temperature (float): Sampling temperature for randomness in responses.
            parallel (bool): Whether to run the requests in parallel.
            n_jobs (int): Number of jobs to run in parallel.

        Returns:
            list: A list of responses from the model.
        """
        if parallel:
            return ParallelTqdm(n_jobs=n_jobs, prefer="threads", total_tasks=len(list_messages))(
                delayed(self.run)(message, temperature=temperature, stream=stream, json_output=json_output) for message in list_messages
            )
        return [self.run(message, temperature) for message in list_messages]

class BaseEmbedding(BaseModel):

    @abc.abstractmethod
    def run(self, prompt: str) -> list[float]:
        pass

    def run_batch(self, prompts: list[str], parallel: bool = False) -> list[list[float]]:
        """
        Generates embeddings for a batch of prompts.

        Args:
            prompts (list[str]): A list of prompts to generate embeddings for.
            parallel (bool): Whether to run the requests in parallel.

        Returns:
            list[list[float]]: A list of generated embedding vectors.
        """
        if parallel:
            return ParallelTqdm(n_jobs=8, prefer="threads", total_tasks=len(prompts))(
                delayed(self.run)(prompt) for prompt in prompts
            )
        return [self.run(prompt) for prompt in tqdm(prompts)]
