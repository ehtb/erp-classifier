import json

from src.boundaries import boundaries
from src.critic import zero_shot_prompt
from src.llm_client import BedrockLLMClient


def test_critic_prompt_uses_the_rule_threshold_and_fills_the_invoice():
    boundary = boundaries("IT_HARDWARE")[0]
    prompt = zero_shot_prompt("INVOICE-123")

    assert f"under {boundary.amount:g} EUR" in prompt
    assert f"{boundary.amount:g} EUR or above" in prompt
    assert "INVOICE-123" in prompt
    assert "{invoice_line}" not in prompt


def test_claude_request_uses_anthropic_messages_api():
    body = BedrockLLMClient._claude_request_body(
        "Classify this invoice", "You are a critic.", 0.0, 32
    )

    assert body["anthropic_version"] == "bedrock-2023-05-31"
    assert body["max_tokens"] == 32
    assert body["temperature"] == 0.0
    assert body["system"] == "You are a critic."
    assert body["messages"] == [{"role": "user", "content": "Classify this invoice"}]
    assert "inferenceConfig" not in body


def test_claude_response_parser_handles_anthropic_shape():
    response = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "GL_6220_IT_PERIPHERALS_OPEX"}],
        "stop_reason": "end_turn",
    }
    assert (
        BedrockLLMClient._claude_response_text(response)
        == "GL_6220_IT_PERIPHERALS_OPEX"
    )


class _FakeBody:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode()


class _FakeBedrockClient:
    def __init__(self, payload):
        self.payload = payload
        self.sent_body = None
        self.model_id = None

    def invoke_model(self, modelId, body):  # noqa: N803 - boto3 keyword name
        self.model_id = modelId
        self.sent_body = json.loads(body)
        return {"body": _FakeBody(self.payload)}


def test_bedrock_generate_dispatches_claude_request_and_response():
    client = object.__new__(BedrockLLMClient)
    client.model_id = "anthropic.claude-sonnet-4-5-20250929-v1:0"
    client.client = _FakeBedrockClient(
        {"content": [{"type": "text", "text": "GL_1400_IT_HARDWARE_CAPEX"}]}
    )

    text = client.generate("Classify this line", system_prompt="Be terse", max_tokens=32)

    assert text == "GL_1400_IT_HARDWARE_CAPEX"
    assert client.client.sent_body["anthropic_version"] == "bedrock-2023-05-31"
    assert client.client.sent_body["system"] == "Be terse"


def test_bedrock_generate_dispatches_nova_request_and_response():
    client = object.__new__(BedrockLLMClient)
    client.model_id = "eu.amazon.nova-pro-v1:0"
    client.client = _FakeBedrockClient(
        {
            "output": {
                "message": {
                    "content": [{"text": "GL_6220_IT_PERIPHERALS_OPEX"}]
                }
            }
        }
    )

    text = client.generate("Classify this line", system_prompt="Be terse", max_tokens=32)

    assert text == "GL_6220_IT_PERIPHERALS_OPEX"
    assert client.client.sent_body["inferenceConfig"]["maxTokens"] == 32
    assert client.client.sent_body["system"] == [{"text": "Be terse"}]


def test_nova_request_and_response_shapes_still_work():
    body = BedrockLLMClient._nova_request_body(
        "Classify this invoice", "You are a critic.", 0.0, 32
    )
    assert body["inferenceConfig"]["maxTokens"] == 32
    assert body["system"] == [{"text": "You are a critic."}]
    assert body["messages"][0]["content"] == [{"text": "Classify this invoice"}]

    response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "GL_1400_IT_HARDWARE_CAPEX"}],
            }
        }
    }
    assert (
        BedrockLLMClient._nova_response_text(response)
        == "GL_1400_IT_HARDWARE_CAPEX"
    )
