import warnings

from langsmith.utils import LangSmithMissingAPIKeyWarning

from .agent_builder import agent_executor_builder

warnings.filterwarnings("ignore", category=LangSmithMissingAPIKeyWarning)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*agent.agent_builder.*"
)

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: F401

from .agent_tools import tools

# load environment variables from .env file
load_dotenv()


def chat(
    initial_input,
    model="gpt-4o",
    provider="openai",
    temperature=0,
    tools=tools,
    prompt_style="structured-chat-agent",
    memory_flag=True,
    handle_parsing_errors=True,
    verbose=False,
):
    """
    Interactive chat function that processes an initial input (if provided) before starting the live conversation.

    Parameters:
        initial_input (str): The input question or prompt provided initially.
        model (str): The model to use. Default is "gpt-4o".
        provider (str): The provider of the model. Default is "openai".
        temperature (float): The temperature setting for the model. Default is 0.
        tools (list): List of tools to use with the agent.
        prompt_style (str): The style of the prompt. Default is "structured-chat-agent".
        memory_flag (bool): Flag for using memory. Default is True.
        handle_parsing_errors (bool): Flag for handling parsing errors. Default is True.
        verbose (bool): Flag for verbose output. Default is False.
    """
    # Define the system prompt.
    system_prompt = (
        "You are a helpful financial assistant. Your task is to answer the user's questions "
        "based on the provided context analyzing carefully the information. If something is not clear, "
        "ask the user for more information. You are not allowed to make up information. If you don't know the answer, "
        "say 'I don't know' and ask the user for more data or details."
    )

    # Create an agent executor with the specified parameters.
    agent_executor, memory = agent_executor_builder(
        model=model,
        provider=provider,
        temperature=temperature,
        tools=tools,
        prompt_style=prompt_style,
        memory_flag=memory_flag,
        handle_parsing_errors=handle_parsing_errors,
        verbose=verbose,
    )

    print("Hi Welcome to the chat! Type 'exit' to end the conversation.")

    # Add the system prompt to the memory.
    memory.chat_memory.add_message(SystemMessage(content=system_prompt))

    # Process initial input if provided
    if initial_input and initial_input.strip():
        memory.chat_memory.add_message(HumanMessage(content=initial_input))
        response = agent_executor.invoke({"input": initial_input})
        print("AI agent:", response["output"])
        memory.chat_memory.add_message(AIMessage(content=response["output"]))

    # Enter the interactive chat loop.
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            print("Exiting the program.")
            break

        memory.chat_memory.add_message(HumanMessage(content=user_input))
        response = agent_executor.invoke({"input": user_input})
        print("AI agent:", response["output"])
        memory.chat_memory.add_message(AIMessage(content=response["output"]))

    print("Thank you for using the chat! Goodbye!")
