import warnings

from langsmith.utils import LangSmithMissingAPIKeyWarning

from .agent_builder import agent_executor_builder

warnings.filterwarnings("ignore", category=LangSmithMissingAPIKeyWarning)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*agent.agent_builder.*"
)
from dotenv import load_dotenv

from .agent_tools import tools
from .prompt_templates import system_prompt  # noqa: F401

# load environment variables from .env file
load_dotenv()


def direct_answer(
    input,
    model="gpt-4o",
    provider="openai",
    temperature=0,
    tools=tools,
    prompt_style="react",
    memory_flag=False,
    handle_parsing_errors=True,
    verbose=True,
):
    """
    Function to get a direct answer from the model using the specified parameters.

    Parameters:
        input (str): The input question or prompt.
        model (str): The model to use. Default is "gpt-4o".
        provider (str): The provider of the model. Default is "openai".
        temperature (float): The temperature setting for the model. Default is 0.
        tools (list): List of tools to use with the agent. Default is an empty list.
        prompt_style (str): The style of the prompt. Default is "react".
        memory_flag (bool): Flag to indicate if memory should be used. Default is False.
        handle_parsing_errors (bool): Flag to indicate if parsing errors should be handled. Default is True.
        verbose (bool): Flag to indicate if verbose output is desired. Default is True.

    Returns:
        str: The response from the model.
    """
    # Create an agent executor with the specified parameters
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

    # Get the response from the agent executor
    response = agent_executor.invoke(
        {
            # "input": f"""Given the following input extract the informations at the index 2702 and then answer the corresponding questions in the input. {model_input}"""
            "input": f"""{system_prompt}. Pay attention to the format it must respect the guidelines. Task: Given the following input extract the informations and then answer the corresponding questions in the input. Do not output 
            any explanantion in the output only the answer {input}"""
        }
    )

    answers = response["output"].split(",")

    return answers
