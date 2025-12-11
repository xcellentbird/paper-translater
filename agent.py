from __future__ import annotations

from typing import Protocol, TypeVar, Generic

import json
import time

from openai import OpenAI
from openai.resources.responses.responses import _type_to_text_format_param
from pydantic import BaseModel

from config import OPENAI_API_KEY


T = TypeVar("T", bound=BaseModel)


class LLMBackend(Protocol):
    """LLM 호출을 담당하는 백엔드 인터페이스 (의존성 역전용)."""

    def parse(self, model: str, messages: list[dict], text_format: type[T]) -> T: ...

    def batch_parse(
        self, model: str, messages_batch: list[list[dict]], text_format: type[T]
    ) -> list[T]: ...


class OpenAILLMBackend:
    """OpenAI Responses / Batches API를 사용하는 실제 백엔드 구현."""

    def __init__(self, client: OpenAI | None = None) -> None:
        self.client = client or OpenAI(api_key=OPENAI_API_KEY)

    def parse(self, model: str, messages: list[dict], text_format: type[T]) -> T:
        response = self.client.responses.parse(
            model=model,
            input=messages,
            text_format=text_format,
            store=False,
        )
        return response.output_parsed

    def batch_parse(
        self, model: str, messages_batch: list[list[dict]], text_format: type[T]
    ) -> list[T]:
        batch_requests = ""
        for i, messages in enumerate(messages_batch):
            request = {
                "custom_id": f"parse_{i}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": model,
                    "input": messages,
                    "text": {"format": _type_to_text_format_param(text_format)},
                    "store": False,
                },
            }
            batch_requests += json.dumps(request) + "\n"

        batch_input_file = self.client.files.create(
            file=batch_requests.encode("utf-8"),
            purpose="batch",
        )

        batch_request = self.client.batches.create(
            completion_window="24h",
            input_file_id=batch_input_file.id,
            endpoint="/v1/responses",
        )

        while True:
            batch_response = self.client.batches.retrieve(batch_request.id)
            batch_status = batch_response.status

            if batch_status == "completed":
                break
            if batch_status == "failed":
                raise RuntimeError(
                    f"Batch failed: {batch_request.id}, {batch_response.errors}"
                )

            time.sleep(0.1)

        batch_output = self.client.files.content(batch_response.output_file_id)
        batch_output = [
            json.loads(line) for line in batch_output.text.split("\n") if line
        ]
        batch_output.sort(key=lambda x: x["custom_id"])

        outputs: list[T] = []
        for output in batch_output:
            output_text = output["response"]["body"]["output"][-1]["content"][0]["text"]
            output_parsed = text_format.model_validate_json(output_text)
            outputs.append(output_parsed)

        return outputs


class LLMAgent(Generic[T]):
    """
    LLM 기반 에이전트.

    - 단일 책임: 에이전트 인터페이스 제공
    - 의존성 역전: 실제 호출은 LLMBackend(예: OpenAILLMBackend)에 위임
    """

    def __init__(self, model: str = "gpt-5.1", backend: LLMBackend | None = None):
        self.model = model
        self.backend: LLMBackend = backend or OpenAILLMBackend()

    def parse(self, messages: list[dict], text_format: type[T]) -> T:
        """
        messages와 text_format을 입력 받아
        LLM 응답을 파싱하여 text_format에 맞는 결과를 반환한다.
        """

        return self.backend.parse(
            model=self.model, messages=messages, text_format=text_format
        )

    def batch_parse(
        self, messages_batch: list[list[dict]], text_format: type[T]
    ) -> list[T]:
        """
        여러 messages 배치를 한 번에 처리하여
        text_format에 맞는 결과 리스트를 반환한다.
        """

        return self.backend.batch_parse(
            model="gpt-5-mini",
            messages_batch=messages_batch,
            text_format=text_format,
        )
