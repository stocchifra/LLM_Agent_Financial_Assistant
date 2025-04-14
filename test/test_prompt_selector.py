from unittest.mock import MagicMock, patch

import pytest
from langchain.agents import (
    create_json_chat_agent,
    create_react_agent,
    create_structured_chat_agent,
    create_tool_calling_agent,
)

from src.agent.prompt_templates import prompt_selector


def test_prompt_selector_valid_styles():
    with patch("src.agent.prompt_templates.hub.pull") as mock_hub:
        mock_prompt = MagicMock()
        mock_hub.return_value = mock_prompt

        # Test ReAct style
        prompt, agent_func = prompt_selector("react")
        assert prompt == mock_prompt
        assert agent_func == create_react_agent

        # Test structured-chat-agent style
        prompt, agent_func = prompt_selector("structured-chat-agent")
        assert prompt == mock_prompt
        assert agent_func == create_structured_chat_agent

        # Test tools-agent style
        prompt, agent_func = prompt_selector("tools-agent")
        assert prompt == mock_prompt
        assert agent_func == create_tool_calling_agent


def test_prompt_selector_json_chat_returns_custom_prompt():
    prompt, agent_func = prompt_selector("json-chat")

    from langchain.prompts.chat import ChatPromptTemplate

    assert isinstance(prompt, ChatPromptTemplate)
    assert agent_func == create_json_chat_agent


def test_prompt_selector_invalid_style_raises_error():
    with pytest.raises(ValueError):
        prompt_selector("invalid-style")
