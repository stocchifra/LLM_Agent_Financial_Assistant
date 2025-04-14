import pytest
from langchain_core.language_models import BaseLanguageModel

# Import the module where agent_builder is defined
from src.agent.agent_builder import agent_builder


# Dummy LLM class to replace ChatOpenAI, ChatAnthropic, ChatVertexAI
class DummyLLM(BaseLanguageModel):
    def __init__(self):
        super().__init__()

    def invoke(self, input):
        return "dummy response"

    # Implement the abstract methods with dummy stubs
    def agenerate_prompt(self, *args, **kwargs):
        return "dummy"

    def apredict(self, *args, **kwargs):
        return "dummy"

    def apredict_messages(self, *args, **kwargs):
        return "dummy"

    def generate_prompt(self, *args, **kwargs):
        return "dummy"

    def predict(self, *args, **kwargs):
        return "dummy"

    def predict_messages(self, *args, **kwargs):
        return "dummy"


# Dummy agent creation function that returns a dummy agent
def dummy_agent_func(llm, tools, prompt):
    return "dummy_agent"


# Dummy prompt_selector function to override the original
def dummy_prompt_selector(prompt_style):
    return "dummy_prompt", dummy_agent_func


@pytest.mark.parametrize(
    "provider, model",
    [
        ("openai", "gpt-4"),
        ("anthropic", "claude-3-sonnet"),
        ("google", "gemini-pro"),
    ],
)
def test_agent_builder(monkeypatch, provider, model):
    # Patch the prompt_selector in the globals of the agent_builder function
    monkeypatch.setitem(
        agent_builder.__globals__, "prompt_selector", dummy_prompt_selector
    )

    # Patch the LLM based on the provider in the globals of agent_builder
    if provider == "openai":
        monkeypatch.setitem(
            agent_builder.__globals__, "ChatOpenAI", lambda **kwargs: DummyLLM()
        )
    elif provider == "anthropic":
        monkeypatch.setitem(
            agent_builder.__globals__, "ChatAnthropic", lambda **kwargs: DummyLLM()
        )
    elif provider == "google":
        monkeypatch.setitem(
            agent_builder.__globals__, "ChatVertexAI", lambda **kwargs: DummyLLM()
        )

    # Call the agent_builder function with dummy values
    agent = agent_builder(
        model=model, provider=provider, temperature=0.0, tools=[], prompt_style="react"
    )

    assert agent == "dummy_agent"
