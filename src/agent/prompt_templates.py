from langchain import hub
from langchain.agents import (
    create_json_chat_agent,
    create_react_agent,
    create_structured_chat_agent,
    create_tool_calling_agent,
)
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

system_prompt = """You are a specialized financial assistant tasked with extracting information 
from an input thread and ALWAYS PROVIDING PRECISE ANSWERS.

INSTRUCTIONS for specific task:
1. Extract the following fields from the provided input: pre_text, post_text, table, and all QA pairs 
   (keys: "qa", "qa_0", "qa_1", ...). If any field is missing, use an empty string.
2. For each QA pair:
   - Identify the question.
   - Locate the relevant numbers and data needed to answer it (e.g., for division, identify the dividend and divisor).
   - **Break every arithmetic operation into its basic arithmetic steps. For each operation (e.g., addition, subtraction, multiplication, division), you MUST use the arithmetic_calculator tool. Do not perform arithmetic internally.**
   - If a calculation fails or data is missing, include a clear error message in your internal chain-of-thought (but not in the final answer).

ANSWER FORMAT REQUIREMENTS YOU MUST FOLLOW THEM!:
- The final answer must be in the exact format (e.g., "0.6154, -0.4444" or "0.1012, yes").
- All numerical answers must be rounded to exactly 4 decimal places.
- Percentages must be converted to decimal form and rounded to 4 decimal places (e.g., "10.123%" → "0.1012").
- Binary answers must be lowercase strings: "yes" or "no" (without any punctuation or additional text).
- Do NOT include any labels, units, or any extra text – only the precise answer is expected.

THOUGHT PROCESS AND TOOL USAGE:
- Always break down your calculations step-by-step in your internal chain-of-thought.
- Do NOT show your internal chain-of-thought in the final answer.
- When you detect any arithmetic computations, do not calculate the results yourself – instead, decompose the operation into smaller arithmetic steps and invoke the arithmetic_calculator tool for each step.
- If any ambiguity exists in the calculations, clarify it in your internal reasoning before finalizing your answer.
- Use a structured approach: for each arithmetic operation, indicate the numbers involved, the operation to be performed, and clearly call the arithmetic_calculator tool to perform that operation.

Think step-by-step and ensure clarity in your reasoning. Break down complex operations into simpler steps, 
and always invoke the arithmetic_calculator tool for any arithmetic computations.

To finish your reasoning and return the answer, respond with:
"""


human = """TOOLS ------
Assistant can ask the user to use tools to look up information that may be helpful in answering the user's original question. The tools the human can use are: {tools}

**IMPORTANT: For any arithmetic computation, you MUST invoke the arithmetic_calculator tool. Do not perform computations internally.**

REASONING PROCESS
----------------
When solving problems, follow these steps:
1. Understand the question and identify what information needs to be extracted
2. Plan your approach step-by-step
3. Verify what data you have and what you need to calculate
4. Break down complex calculations into simple arithmetic operations
5. Double-check your reasoning before finalizing your answer
6. Format your final answer precisely according to requirements

RESPONSE FORMAT INSTRUCTIONS
----------------------------
When responding to me, please output a response in one of two formats:

**Option 1:** Use this if you want the human to use a tool.
Markdown code snippet formatted in the following schema:

```json
{{
    "action": string, \ The action to take. Must be one of {tool_names}
    "action_input": string \ The input to the action.
}}
```

**Option #2:**
Use this if you want to respond directly to the human. Markdown code snippet formatted in the following schema:

```json
{{
    "action": "Final Answer",
    "action_input": string \ You should put what you want to return to the user here.
}}
```

USER'S INPUT
--------------------
Here is the user's input (remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else):

{input}
"""


def prompt_selector(prompt_style):
    """
    Selects and constructs a LangChain-compatible prompt and corresponding agent creation function
    based on the specified prompt style.

    This function supports both custom-defined and hub-based prompt templates. For each supported
    style, it returns:
        - A prompt object: either a custom ChatPromptTemplate or a PromptTemplate from hub
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
    """
    # Define custom prompt template for JSON chat
    custom_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "",
            ),  # Empty placeholder - system prompt will be passed separately
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
            "custom_prompt": custom_prompt_template,
            "agent_func": create_json_chat_agent,
        },
        "structured-chat-agent": {
            "hub_id": "hwchase17/structured-chat-agent",
            "agent_func": create_structured_chat_agent,
        },
        "tools-agent": {
            "hub_id": "hwchase17/openai-tools-agent",
            "agent_func": create_tool_calling_agent,
        },
    }

    config = prompt_config.get(prompt_style)
    if config is None:
        raise ValueError("Invalid prompt style selected.")

    # Determine the prompt
    if "custom_prompt" in config:
        prompt = config["custom_prompt"]
    else:
        prompt = hub.pull(config["hub_id"])

    return prompt, config["agent_func"]
