import json
import warnings

from dotenv import load_dotenv
from langchain.agents import (
    AgentExecutor,
)
from langchain_core._api.deprecation import LangChainDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
# from .. import warnings_config # noqa: F401
import signal

from langchain.memory import ConversationBufferMemory
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: F401
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI

from src.utils import extract_selected_threads_processed, get_exact_answers

from .agent_tools import tools
from .prompt_templates import prompt_selector, system_prompt  # noqa: F401


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Agent execution took too long!")


# load environment variables from .env file
load_dotenv()


def agent_builder(model, provider, temperature, tools, prompt_style):
    """
    Constructs and returns a LangChain AgentExecutor configured with the specified LLM model, provider,
    toolset, and prompt style.

    This function:
        - Instantiates the appropriate LLM wrapper (OpenAI, Anthropic, or Google Vertex) based on the
          model and provider.
        - Selects the appropriate prompt template and agent creation function via the `prompt_selector`.
        - Builds a custom agent using the provided tools and the selected prompt.
        - Wraps the agent in an AgentExecutor to handle execution, error handling, and step tracking.

    Parameters:
        model (str): The name of the model to use (e.g., "gpt-4", "claude-3-sonnet", "gemini-pro").
        provider (str): The LLM provider ("openai", "anthropic", "google").
        temperature (float): Sampling temperature for the model output.
        tools (list): A list of tools (e.g., functions or plugins) the agent can use during execution.
        prompt_style (str): The prompt configuration style, passed to `prompt_selector`
                            (e.g., "react", "json-chat", etc.).

    Returns:
        AgentExecutor: A fully configured agent executor ready for task execution.

    Raises:
        ValueError: If the provider is not recognized.
    """

    # Initialize the LLM based on the provider.
    if provider == "openai" and model.startswith("gp"):
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
        )
    elif provider == "openai" and model.startswith("o"):
        llm = ChatOpenAI(
            model=model,
        )
    elif provider == "anthropic":
        llm = ChatAnthropic(
            model=model,
            temperature=temperature,
            # temperature=1,
            # max_tokens=6000,
            # thinking={"type": "enabled", "budget_tokens": 5000}
        )

    elif provider == "google":
        llm = ChatVertexAI(
            model=model,
            temperature=temperature,
        )
    else:
        raise ValueError("Invalid provider selected.")

    # Select the prompt based on the provided style.
    prompt, agent_func = prompt_selector(prompt_style)

    # Create the agent using the mapped function.
    agent = agent_func(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    return agent


def agent_executor_builder(
    model,
    provider,
    temperature,
    tools,
    prompt_style,
    memory_flag=False,
    handle_parsing_errors=True,
    verbose=True,
):

    # Initilize the Agent
    agent = agent_builder(
        model=model,
        provider=provider,
        temperature=temperature,
        tools=tools,
        prompt_style=prompt_style,
    )

    # create an agent executor with the agent and tools
    if memory_flag:
        # cnversation buffer memory creation
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="output",
            return_messages=True,
        )

        # add system prompt to memory
        # memory.chat_memory.add_message(SystemMessage(content=system_prompt))
        # Register the handler
        # signal.signal(signal.SIGALRM, timeout_handler)
        # signal.alarm(45)  # set timeout (e.g. 30 seconds)

        # create an agent executor with the agent and tools
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=verbose,
            max_iterations=10,
            return_intermediate_steps=True,
            handle_parsing_errors=handle_parsing_errors,
            memory=memory,
        )
        return agent_executor, memory

    else:
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=verbose,
            max_iterations=10,
            return_intermediate_steps=True,
            handle_parsing_errors=handle_parsing_errors,
        )

        return agent_executor, None
