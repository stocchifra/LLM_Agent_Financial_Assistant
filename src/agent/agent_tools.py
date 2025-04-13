import json
import os
from typing import Union

from langchain_core.tools import Tool


def extract_thread_details_fun(thread):
    """
    Extract required fields from a single thread.
    Parameters:
    thread (dict): A dictionary representing a JSON thread that may contain the keys:
    'pre_text', 'post_text', 'table_ori', 'table', 'id', 'exe_ans',
    and one or more QA fields such as 'qa', 'qa_0', 'qa_1'.
    Returns:
    dict: A dictionary with extracted thread details
    """
    # Initialize the output dictionary with the defined keys.
    thread_extracted = {
        "pre_text": thread.get("pre_text", []),
        "post_text": thread.get("post_text", []),
        # "table_ori": thread.get("table_ori", []),
        "table": thread.get("table", []),
        # "id": thread.get("id", ""),
        "qa": [],  # will be a list of QA dictionaries
    }

    # Look for keys that are exactly "qa" or start with "qa_" (e.g., "qa_0", "qa_1", etc.)
    for key in thread:
        if key == "qa" or key.startswith("qa_"):
            qa_data = thread.get(key)
            if isinstance(qa_data, dict):
                qa_dict = {
                    # "qa_field": key,
                    "question": qa_data.get("question", None),
                    # "exe_ans": qa_data.get("exe_ans", None), # Extract the exe_ans for each corresponding QA.
                }
                thread_extracted["qa"].append(qa_dict)

    return thread_extracted


def extract_all_threads_fun(data):
    """
    Process a list of JSON threads and extract specified keys from each one.
    Parameters:
    data (list): List of thread dictionaries loaded from a JSON file.
    Returns:
    list: A list of dictionaries with the extracted information.
    """
    return [extract_thread_details_fun(thread) for thread in data]


def extract_financial_data(input_data) -> str:
    """
    Extract financial data from the input, which can be:
    - A file path to a JSON file,
    - A JSON-formatted string, or
    - An already loaded JSON object (list or dictionary).

    Args:
        input_data: A string representing either a file path to a JSON file or a JSON-formatted string,
                    or already loaded JSON (list or dict).

    Returns:
        A JSON-formatted string representing the extracted financial information.
    """
    try:
        # If input is already a loaded JSON (list or dict), use it directly.
        if isinstance(input_data, (list, dict)):
            data = input_data
        # If input is a string, determine if it is a file path or a JSON string.
        elif isinstance(input_data, str):
            if os.path.exists(input_data):
                # The string is a valid file path, so load the JSON data from the file.
                with open(input_data, "r") as f:
                    data = json.load(f)
            else:
                # The string is not a file path; assume it is a JSON-formatted string.
                data = json.loads(input_data)
        else:
            raise ValueError(
                "Invalid input type. Provide a file path, a JSON string, or loaded JSON data."
            )

        # Depending on the type of JSON data (a single thread as dict or multiple threads as list),
        # call the proper processing function.
        if isinstance(data, list):
            financial_data = extract_all_threads_fun(data)
        elif isinstance(data, dict):
            financial_data = extract_thread_details_fun(data)
        else:
            raise ValueError("The provided JSON data is neither a list nor a dict.")

        return json.dumps(financial_data, indent=2)
    except Exception as e:
        return json.dumps(
            {"error": f"Failed to extract financial data: {str(e)}"}, indent=2
        )


def perform_math_calculus(expression: str) -> Union[float, str]:
    """
    Perform basic arithmetic operations based on the provided expression.

    Args:
        expression (str): The mathematical expression to evaluate.

    Returns:
        float: The result of the arithmetic operation.
    """
    try:
        # Remove quotes if present
        cleaned_expr = expression.strip().strip('"').strip("'")
        result = eval(cleaned_expr)
        return str(round(result, 4))
    except Exception as e:
        return f"Error in calculation: {e}"


# List of tools to be used in the agent
tools = [
    Tool(
        name="arithmetic_calculator",
        description="""Performs arithmetic calculations based on the provided expression. 
        Input: Mathematical expression as a string (e.g., "2 + 2", "10 / 3", "5 * 6", "(8 - 4) / 2", "5 ** 2").
        Output: Result of the arithmetic operation, rounded to 4 decimal places.
        Example output:
        - For input "2 + 2", output: 4.0000
        - For input "10 / 3", output: 3.3333
        """,
        func=perform_math_calculus,
    ),
    Tool(
        name="financial_data_extraction",
        description="""
        Extracts structured financial information from context using `extract_financial_data`.

        **Input**:
        - A string representing:
        - A file path to a JSON file,
        - A JSON-formatted string,
        - Or an already loaded JSON object (list or dict) containing financial thread data.

        **Output**:
        - A JSON-formatted string that includes these keys:
        - "pre_text": A list of text paragraphs that appear before the table,
        - "post_text": A list of text paragraphs that appear after the table,
        - "table": A list of lists representing tabular data; the first sub-list typically contains column identifiers,
        - "id": Document identifier (e.g., "Single_JKHY/2009/page_28.pdf-3"),
        - "qa": A list of dictionaries containing at least the question(s) to be answered.

        **Example Output**:
        ```json
        {
        "pre_text": [
            "Paragraph about financial performance...",
            "Details about revenue..."
        ],
        "post_text": [
            "Information about cash flow...",
            "Details about fiscal year..."
        ],
        "table": [
            ["2008", "year ended june 30 2009 2008", "year ended june 30 2009 2008"],
            ["net income", "$ 103102", "$ 104222", "$ 104681"],
            ["non-cash expenses", "74397", "70420", "56348"]
        ],
        "qa": [
            {
            "question": "what was the percentage change in the net cash from operating activities from 2008 to 2009"
            }
        """,
        func=extract_financial_data,
    ),
]
