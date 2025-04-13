import json

from agent_tools import extract_financial_data, tools
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import initialize_agent  # noqa: F401
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)
from langchain_core.tools import Tool  # noqa: F401
from langchain_openai import ChatOpenAI
from prompt_templates import prompt_template  # noqa: F401

# load environment variables from .env file
load_dotenv()

# select the prompt
prompt = hub.pull("hwchase17/react")


# Initialize the LLM with the specified model and temperature
llm = ChatOpenAI(
    # model="gpt-4o-mini-2024-07-18",
    model="gpt-4o",
    temperature=0,
)

# Create a ReAct agent with the LLM and prompt template
agent = create_react_agent(
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


if __name__ == "__main__":
    data_path = "/Users/francescostocchi/ConvFinQA_LLM_Project/data/train.json"
    # Open the JSON file and load the data.
    with open(data_path, "r") as f:
        data = json.load(f)

    # print(type(data))

    single_sample = data[90]
    exact_answers = []
    for key in single_sample:
        if key == "qa" or key.startswith("qa_"):
            qa_data = single_sample.get(key)
            if isinstance(qa_data, dict):
                exact_answers.append(qa_data.get("exe_ans", None))

    print("Exact answers:", exact_answers)

    model_input = extract_financial_data(single_sample)

    response = agent_executor.invoke(
        {
            "input": f"""Given the input extract the relevant data and then answer all the questions {model_input}.
        If the answer is a number, output it with 4 decimal places.
        If it's a percentage, convert to decimal with 4 decimal places (e.g. 10.123% -> 0.1012).
        If it's a binary answer, output only yes or no.
        Output only the final precise answers in the answer field.
        Do not include any additional explanations or commentary outside of the thought process field."""
        }
    )

    print(type(response))
    # Split the 'output' string by comma and strip whitespace
    answers = response["output"].split(",")
    print("Answers:", answers)
