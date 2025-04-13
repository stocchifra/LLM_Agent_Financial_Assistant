import json
import os
from typing import Union

from langchain_core.tools import Tool

from utils import extract_selected_threads, open_json_file


def extract_thread_details_fun(thread):
    """
    Extract required fields from a single thread.
    Parameters:
      thread (dict): A dictionary representing a JSON thread that may contain various keys.
    Returns:
      dict: A dictionary with extracted thread details.
    """
    thread_extracted = {
        "pre_text": thread.get("pre_text", []),
        "post_text": thread.get("post_text", []),
        "table": thread.get("table", []),
        "qa": [],
    }
    for key in thread:
        if key == "qa" or key.startswith("qa_"):
            qa_data = thread.get(key)
            if isinstance(qa_data, dict):
                qa_dict = {"question": qa_data.get("question", None)}
                thread_extracted["qa"].append(qa_dict)
    return thread_extracted


# def extract_selected_threads(data, indexes=None):
#     """
#     Process a list of JSON threads and extract specified keys from each one.
#     Parameters:
#       data (list): List of thread dictionaries loaded from a JSON file.
#       indexes: None, a tuple (start, end), or a list of specific indices.
#     Returns:
#       list: A list of dictionaries with the extracted information.
#     """
#     if indexes is None:
#         return [extract_thread_details_fun(thread) for thread in data]
#     elif isinstance(indexes, list):
#         return [extract_thread_details_fun(data[i]) for i in indexes if i < len(data)]
#     elif isinstance(indexes, tuple):
#         start, end = indexes
#         if end > len(data):
#             raise IndexError("Index out of range")
#         return [extract_thread_details_fun(thread) for thread in data[start:end]]


def extract_financial_data(input_data):
    """
    Extract financial data from the input, which can be provided as a string.

    Accepted input formats:
    1. A JSON-formatted string representing a dictionary with the following structure:
         {
           "tool_input": {
             "file_path": "Path/to/your/file.json",
             "indexes": <optional_indexes_parameter>
           }
         }
       - file_path: Can be a file path to a JSON file, a JSON-formatted string,
         or an already loaded JSON object (list or dict) containing financial thread data.
       - indexes: (Optional) Allows you to specify a subset of threads. It can be:
           - None: Process the entire input.
           - A tuple: (start, end) to extract a contiguous slice of threads.
             For example, (90, 91) extracts only the thread at index 90.
           - A list: e.g., [3, 7, 15] to extract only those specific threads.

    2. A plain string representing a file path or a JSON-formatted string.

    Returns:
      A dictionary representing the extracted financial information.
    """
    try:
        # Since the LLM always outputs a string, first try to parse it as JSON.
        parsed = None
        try:
            parsed = json.loads(input_data)
        except Exception:
            parsed = None

        if parsed is not None and isinstance(parsed, dict) and "tool_input" in parsed:
            tool_input = parsed["tool_input"]
            file_path = tool_input.get("file_path")
            indexes = tool_input.get("indexes", None)
            if not file_path:
                raise ValueError("File path is required in the tool_input")
            # Ensure file_path is a string; if it is not, attempt to convert it.
            if not isinstance(file_path, str):
                # If file_path is a dict, try to extract a string representation from a known key.
                if isinstance(file_path, dict) and "file" in file_path:
                    file_path = file_path["file"]
                else:
                    file_path = str(file_path)
            if os.path.exists(file_path):
                # with open(file_path, "r") as f:
                #     data = json.load(f)
                data = open_json_file(file_path)
            else:
                # If the file doesn't exist, assume file_path holds the actual JSON data.
                data = json.loads(file_path)
        else:
            # If parsing as JSON didn't return a structured dict, treat input_data as a file path or raw JSON string.
            if os.path.exists(input_data):
                with open(input_data, "r") as f:
                    data = json.load(f)
                indexes = None
            else:
                data = json.loads(input_data)
                indexes = None

        financial_data = extract_selected_threads(data, indexes)
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
    Tool(
        name="financial_data_extraction",
        description="""
        Extracts structured financial information from context using the function `extract_financial_data`.

        **Important Note about File Paths**:
        - The `file_path` field in `tool_input` must be a valid path to an existing JSON file (e.g., "data/financial_threads.json"). 
        - Do **not** provide a lengthy passage of text or a PDF reference as the `file_path`; the tool will fail if it cannot locate the file on disk or parse its content as JSON.
        - If you provide a JSON object/string directly instead of a file path, the tool will ignore the `indexes` parameter and process the entire data.

        **Backup Plan**:
        - If the tool fails to parse the file or JSON data, it will attempt a manual extraction of any date patterns (in the format YYYY-MM-DD) from the provided input string. If no date is found, it will return the original error message.

        **Input**:
        - A JSON-formatted string representing a dictionary with the following structure:
        {
            "tool_input": {
            "file_path": "Path/to/your/file.json",
            "indexes": <optional_indexes_parameter>
            }
        }
        where:
        - `file_path`: A string pointing to a JSON file (e.g., "data/financial_threads.json"), or a JSON object/string to be processed directly (in which case `indexes` is ignored).
        - `indexes` (optional): A subset of threads to process (tuple or list). Ignored if `file_path` is a direct JSON string/object.

        **Output**:
        - A dictionary containing:
        - "pre_text": A list of text paragraphs that appear before the table,
        - "post_text": A list of text paragraphs that appear after the table,
        - "table": A list of lists representing tabular data (the first sub-list typically contains column identifiers),
        - "qa": A list of dictionaries containing at least the question(s) to be answered.
        - If an unrecoverable error occurs, a dictionary with an "error" field is returned.
        - If the tool cannot parse the file but finds date patterns in the input, it returns them in the "extracted_dates" list.

        **Usage Examples**:

        - **Extract the entire JSON file (all threads)**:
        result = extract_financial_data('{"tool_input": {"file_path": "data/financial_threads.json"}}')

        - **Extract a single thread at a specific index (e.g., index 90)**:
        result = extract_financial_data('{"tool_input": {"file_path": "data/financial_threads.json", "indexes": [90, 91]}}')

        - **Extract a range of threads (e.g., threads from index 10 to 19)**:
        result = extract_financial_data('{"tool_input": {"file_path": "data/financial_threads.json", "indexes": [10, 20]}}')

        - **Extract specific threads using a list of indices (e.g., indices 3, 7, and 15)**:
        result = extract_financial_data('{"tool_input": {"file_path": "data/financial_threads.json", "indexes": [3, 7, 15]}}')

        - **Provide a JSON object/string directly** (ignoring `indexes`):
        result = extract_financial_data('{"pre_text": ["Example text"], "post_text": ["More text"], "table": [[...]], "qa": [{"question": "Example?"}]}')
        """,
        func=extract_financial_data,
    ),
]
