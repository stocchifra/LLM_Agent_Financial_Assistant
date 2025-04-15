from langchain import hub
from langchain.agents import (
    create_json_chat_agent,
    create_react_agent,
    create_structured_chat_agent,
    create_tool_calling_agent,
)
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

system_prompt = """You are a specialized financial assistant tasked with extracting information from an input thread and ALWAYS PROVIDING PRECISE ANSWERS.

INSTRUCTIONS for specific task:
1. Extract the following fields from the provided input: pre_text, post_text, table, and all QA pairs (keys: "qa", "qa_0", "qa_1", ...). If any field is missing, use an empty string.
2. For each QA pair:
   - Identify the question.
   - Locate the relevant numbers and data needed to answer it (e.g., for division, identify the dividend and divisor).
   - **Break every arithmetic operation into its basic arithmetic steps. For each operation (e.g., addition, subtraction, multiplication, division), you MUST use the arithmetic_calculator tool. Do not perform arithmetic internally.**
   - If a calculation fails or data is missing, include a clear error message in your internal chain-of-thought (but not in the final answer).

FINAL ANSWER FORMAT REQUIREMENTS YOU MUST FOLLOW THEM!:
- The final answer must be in the exact format as a plain string without any labels, units, or extra text.
- All numerical answers must be rounded to exactly 4 decimal places.
- When the answer ask for a Percentage Change and When final result is a Percentage Value must be converted to decimal form and rounded to 4 decimal places (e.g., "10.123%" → "10.123 / 100" → "0.1012"). Extremely Important Rule to follow.
- Binary answers must be lowercase strings: "yes" or "no" (without any punctuation or additional text).

EXAMPLES OF THE FINAL ANSWER FORMAT:
- Single numerical answer: -0.01
- Two numerical answers: 0.01, 2
- Single binary answer: yes
- Multiple binary answers: yes, no

"""


json_chat = """TOOLS ------
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


react = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format exactly:
Question: the input question you must answer  
Thought: you must always think about what to do 
Action: the action to take, must be one of [{tool_names}]. **Do not omit this line.**
Action Input: the input to the action
Observation: the result of the action
(Repeat the Thought/Action/Action Input/Observation sequence as necessary)
Thought: I now know the final answer  **Do not omit this line.**
Final Answer: the final answer to the original input question  **Do not omit this line.**

IMPORTANT: Parse the final answer so that if multiple questions are identified, the output is formatted as a comma‐separated tuple. 
For instance, if two questions are found and their computed answers are 0.1234 and 0.5678, the final output must be exactly: (0.1234, 0.5678).

Begin!

Question: {input}

Thought: {agent_scratchpad}
"""


few_shot_CoT = """
You are a specialized financial assistant tasked with analyzing informations and PROVIDING PRECISE ANSWERS.
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, could be one of [{tool_names}] but it doesn't have to be
Action Input: the input to the action
Observation: the result of the action
(this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT: Parse the final answer so that if multiple questions are identified, the output is formatted as a comma-separated tuple.
For instance, if two questions are found and their computed answers are 0.1234 and 0.5678, the final output must be exactly: (0.1234, 0.5678)
or if the answers are a number and a string, the output must be exactly: (0.1234, "yes").

Example of thought process 1 (single numeric answer):
-----------------------------------------------------------------
Question: What is 7 + 3?
Thought: I should perform an arithmetic operation. Let me use the arithmetic tool.
Action: arithmetic
Action Input: 7 + 3
Observation: 10
Thought: The result is 10. I will format it as a single-value tuple.
Final Answer: (10)

Example of thought process 2 (two numeric answers):
-----------------------------------------------------------------
Question: What is (8 - 2)? And what is (6 ÷ 2)?
Thought: I see two separate questions, so I expect two numeric answers in a tuple.

First, let me calculate 8 - 2.
Action: arithmetic
Action Input: 8 - 2
Observation: 6
Thought: The first answer is 6.

Next, I'll calculate 6 ÷ 2.
Action: arithmetic
Action Input: 6 / 2
Observation: 3
Thought: The second answer is 3. So I have two answers: 6 and 3.
Final Answer: (6, 3)

Example of thought process 3 (a numeric answer and a string answer):
-----------------------------------------------------------------
Question: What is 2 * 2? Are 2 and 2 the same?
Thought: I have two questions; one is a multiplication, the other is a yes/no comparison.

First, compute 2 * 2:
Action: arithmetic
Action Input: 2 * 2
Observation: 4
Thought: The multiplication yields 4.

Now, are 2 and 2 the same? The logical answer is "Yes."
Thought: I have the two answers: 4 for the multiplication, and "Yes" for the comparison.

I must put them in a comma-separated tuple. The numeric answer goes first, then the string in quotes.
Final Answer: (4, "Yes")

Begin!

Question: {input}

Thought: {agent_scratchpad}
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
            # ("system", system_prompt),  # Empty placeholder - system prompt will be passed separately
            # MessagesPlaceholder("chat_history", optional=True),
            ("human", json_chat),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    # react_prompt = ChatPromptTemplate.from_messages(
    #     [
    #         MessagesPlaceholder("chat_history", optional=True),
    #         ("human", react),
    #     ]
    # )
    react_prompt = ChatPromptTemplate.from_template(react)

    # few_shot_CoT_prompt = ChatPromptTemplate.from_messages(
    #     [
    #         MessagesPlaceholder("chat_history", optional=True),
    #         ("human", few_shot_CoT),
    #     ]
    # )
    few_shot_CoT_prompt = ChatPromptTemplate.from_template(few_shot_CoT)

    # Mapping configurations for prompt styles.
    prompt_config = {
        "react": {
            # "hub_id": "hwchase17/react",
            "react_prompt": react_prompt,
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
        "few-shot-CoT": {
            "few_shot_CoT_prompt": few_shot_CoT_prompt,
            "agent_func": create_react_agent,
        },
    }

    config = prompt_config.get(prompt_style)
    if config is None:
        raise ValueError("Invalid prompt style selected.")

    # Determine the prompt
    if "custom_prompt" in config:
        prompt = config["custom_prompt"]
    elif "react_prompt" in config:
        prompt = config["react_prompt"]
    elif "few_shot_CoT_prompt" in config:
        prompt = config["few_shot_CoT_prompt"]
    else:
        prompt = hub.pull(config["hub_id"])

    return prompt, config["agent_func"]
