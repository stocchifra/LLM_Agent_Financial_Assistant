from langchain import hub
from langchain.agents import (
    create_json_chat_agent,
    create_react_agent,
    create_structured_chat_agent,
    create_tool_calling_agent,
)
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def prompt_selector(prompt_style):
    """
    Selects and constructs a LangChain-compatible prompt and corresponding agent creation function
    based on the specified prompt style.

    This function supports both custom-defined and hub-based prompt templates. For each supported
    style, it returns:
        - A prompt object: either a custom ChatPromptTemplate or a PromptTemplate constructed
          from a fixed system prompt + a pulled hub prompt.
        - The associated agent creation function for that style.

    Parameters:
        prompt_style (str): The identifier for the desired prompt style.
        Must be one of: "react", "json-chat", "structured-chat-agent", "tools-agent".

    Returns:
        tuple: (prompt, agent_func)
            - prompt: a LangChain prompt template object
            - agent_func: a function to create the corresponding agent

    Raises:
        ValueError: If the provided prompt_style is not recognized.

    Note:
        - "json-chat" uses a fully custom prompt template.
        - Other styles fetch predefined templates from LangChain hub and prepend a consistent
          system prompt for task-specific instructions.
    """

    system_prompt = """You are a specialized financial assistant tasked with extracting information 
    from an input thread and providing precise answers.

    INSTRUCTIONS for specific task:
    - First, recognize and extract the following fields from the provided input: pre_text, post_text, table, 
    and all QA pairs (keys: "qa", "qa_0", "qa_1", ...). If any field is missing, default to an empty string.
    - For each QA pair:
        1. Identify the question.
        2. From the extracted data, locate the numbers required to answer the question (e.g., for division, 
        identify the dividend and divisor) and, if needed, elaborate on the mathematical function.
        3. If a calculation is needed, invoke the arithmetic tool accordingly. If the arithmetic tool fails, 
        provide a clear error message.
    - When providing an answer:
        - For binary answers, output only "yes" or "no."
        - For numerical answers, output a number with exactly 4 decimal places.
        - For percentages, convert them to decimal form with exactly 4 decimal places 
        (e.g., 10.123% → 0.1012).

    Do not include any additional explanations or commentary outside of the "thought process" field.
    Only output the final precise answers in the "answer" field."""

    human = '''TOOLS
    ------
    Assistant can ask the user to use tools to look up information that may be helpful in answering the users original question. The tools the human can use are:

    {tools}

    RESPONSE FORMAT INSTRUCTIONS
    ----------------------------

    When responding to me, please output a response in one of two formats:

    **Option 1:**
    Use this if you want the human to use a tool.
    Markdown code snippet formatted in the following schema:

    """json
    {{
        "action": string, \ The action to take. Must be one of {tool_names}
        "action_input": string \ The input to the action
    }}
    """

    **Option #2:**
    Use this if you want to respond directly to the human. Markdown code snippet formatted in the following schema:

    """json
    {{
        "action": "Final Answer",
        "action_input": string \ You should put what you want to return to use here
    }}
    """

    USER'S INPUT
    --------------------
    Here is the user's input (remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else):

    {input}'''

    custom_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", human),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    # Mapping configurations for prompt styles.
    prompt_config = {
        "react": {
            "hub_id": "hwchase17/react",
            "agent_func": create_react_agent,
        },
        "json-chat": {
            # For json-chat, we assume you want to use a custom prompt template already defined.
            "custom_prompt": custom_prompt_template,
            "agent_func": create_json_chat_agent,  # Ensure create_json_chat_agent is imported/defined.
        },
        "structured-chat-agent": {
            "hub_id": "hwchase17/structured-chat-agent",
            "agent_func": create_structured_chat_agent,
        },
        "tools-agent": {
            "hub_id": "hwchase17/openai-tools-agent",
            "agent_func": create_tool_calling_agent,  # Adjust this based on availability.
        },
    }

    config = prompt_config.get(prompt_style)
    if config is None:
        raise ValueError("Invalid prompt style selected.")

    # Determine the hub prompt.
    if "custom_prompt" in config:
        prompt = config["custom_prompt"]
    else:
        hub_prompt = hub.pull(config["hub_id"])
        # If hub_prompt is a PromptTemplate, extract the underlying string.
        if hasattr(hub_prompt, "template"):
            hub_prompt_str = hub_prompt.template
        else:
            hub_prompt_str = hub_prompt

        # Combine your custom system prompt with the hub prompt string.
        combined_prompt_str = system_prompt + "\n\n" + hub_prompt_str

        # Wrap the combined prompt string in a PromptTemplate.
        # Adjust the input_variables as needed. Here we assume it needs an "input" variable.
        prompt = PromptTemplate(template=combined_prompt_str, input_variables=["input"])

    return prompt, config["agent_func"]
