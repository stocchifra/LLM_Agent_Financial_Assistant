import json
from typing import Any, Dict, Union

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


def extract_financial_data(input_str: str) -> Dict[str, Any]:
    """
    Extract financial data from input JSON string or dictionary

    Args:
        input_str: JSON string or dictionary containing financial thread data

    Returns:
        Dictionary with extracted financial information
    """
    try:
        # Handle the case where input might be a string or already a dictionary/list
        if isinstance(input_str, str):
            data = json.loads(input_str)
        else:
            data = input_str

        # Process single thread or list of threads accordingly
        if isinstance(data, list):
            financial_data = extract_all_threads_fun(data)
        else:
            financial_data = extract_thread_details_fun(data)
        return financial_data
    except Exception as e:
        return {"error": f"Failed to extract financial data: {str(e)}"}


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
    # Tool(
    #     name="financial_data_extraction",
    #     description="""Extracts structured financial information from context.
    #     Input: JSON string or dictionary containing financial thread data.
    #     Output: Dictionary with these keys:
    #     - pre_text: List of text paragraphs appearing before the table
    #     - post_text: List of text paragraphs appearing after the table
    #     - table: List of lists representing tabular data, where the first list typically contains column identifiers
    #     - id: Document identifier string (e.g., 'Single_JKHY/2009/page_28.pdf-3')
    #     - qa: List of dictionaries containing only the question to be answered
    #     Example output structure:
    #     {
    #         "pre_text": ["Paragraph about financial performance...", "Details about revenue..."],
    #         "post_text": ["Information about cash flow...", "Details about fiscal year..."],
    #         "table": [
    #             ["2008", "year ended june 30 2009 2008", "year ended june 30 2009 2008"],
    #             ["net income", "$ 103102", "$ 104222", "$ 104681"],
    #             ["non-cash expenses", "74397", "70420", "56348"]
    #         ],
    #         "qa": [{"question": "what was the percentage change in the net cash from operating activities from 2008 to 2009"}]
    #     }
    #     """,
    #     func=extract_financial_data,
    # )
]
