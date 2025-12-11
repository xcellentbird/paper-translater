from agent import LLMAgent, LLMBackend
import pytest
from pydantic import BaseModel


class FakeBackend(LLMBackend):
    """실제 OpenAI 호출 대신, 단순한 더미 응답을 만드는 백엔드."""

    def parse(self, model: str, messages: list[dict], text_format: type[BaseModel]):
        text = " ".join(str(m.get("content", "")) for m in messages)
        return text_format(text=text)

    def batch_parse(
        self,
        model: str,
        messages_batch: list[list[dict]],
        text_format: type[BaseModel],
    ):
        results: list[BaseModel] = []
        for messages in messages_batch:
            text = " ".join(str(m.get("content", "")) for m in messages)
            results.append(text_format(text=text))
        return results


@pytest.fixture
def agent():
    # 테스트에서는 네트워크를 사용하지 않도록 FakeBackend를 주입한다.
    return LLMAgent(model="test-model", backend=FakeBackend())


def test_parse(agent):
    messages = [{"role": "user", "content": "Hello, how are you?"}]

    class Response(BaseModel):
        """
        Response model
        """

        text: str

    response = agent.parse(messages, text_format=Response)
    assert isinstance(response, Response)
    assert "Hello, how are you?" in response.text


def test_batch_parse(agent):
    messages_batch = [
        [{"role": "user", "content": "Hey, What's up?"}],
        [{"role": "user", "content": "What's your name?"}],
    ]

    class Response(BaseModel):
        """
        Response model
        """

        text: str

    responses = agent.batch_parse(messages_batch, text_format=Response)
    assert len(responses) == 2
    assert isinstance(responses[0], Response)
    assert isinstance(responses[1], Response)
    assert "Hey, What's up?" in responses[0].text
    assert "What's your name?" in responses[1].text
