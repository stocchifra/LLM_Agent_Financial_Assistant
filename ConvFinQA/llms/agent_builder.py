import json

from agent_tools import extract_financial_data, tools
from dotenv import load_dotenv
from langchain.agents import (
    AgentExecutor,
)
from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
from prompt_templates import prompt_selector  # noqa: F401

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

    # create an agent executor with the agent and tools
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )

    return agent_executor


if __name__ == "__main__":
    data_path = "/Users/francescostocchi/ConvFinQA_LLM_Project/data/train.json"
    # Open the JSON file and load the data.
    with open(data_path, "r") as f:
        data = json.load(f)

    # print(type(data))

    single_sample = data[0]
    exact_answers = []
    for key in single_sample:
        if key == "qa" or key.startswith("qa_"):
            qa_data = single_sample.get(key)
            if isinstance(qa_data, dict):
                exact_answers.append(qa_data.get("exe_ans", None))

    print("Exact answers:", exact_answers)

    model_input = extract_financial_data(single_sample)
    # model_input = "/Users/francescostocchi/ConvFinQA_LLM_Project/data/test_private.json"

    agent_executor = agent_builder(
        model="gpt-4o",
        # model="o3-mini",
        provider="openai",
        temperature=0,
        tools=tools,
        # prompt_style="tools-agent",
        prompt_style="react",
        # prompt_style="structured-chat-agent",
        # prompt_style="json-chat",
    )

    response = agent_executor.invoke(
        {
            "input": f"""Given the input extract the relevant data and then answer all the questions {model_input}."""
        }
    )

    # print(response)
    # Split the 'output' string by comma and strip whitespace
    answers = response["output"].split(",")
    intermediate_steps = response["intermediate_steps"]
    print("Intermediate steps:", intermediate_steps)
    print("Answers:", answers)
