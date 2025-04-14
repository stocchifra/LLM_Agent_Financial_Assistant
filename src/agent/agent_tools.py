import ast
import json
import os
import re
from typing import Union

from langchain_core.tools import Tool

from src.utils import open_json_file


def strip_code_fence(input_str: str) -> str:
    """
    Removes common multiline string delimiters from the input string.

    This function detects and removes:
      - Triple backticks (```), including those with a language indicator (e.g., ```json)
      - Triple double quotes, including ones with a language indicator
      - Triple single quotes ('''...''')

    If none of these delimiters are detected, the string is returned as-is.
    """
    # Normalize newlines to \n
    input_str = input_str.replace("\r\n", "\n").strip()

    # List of regex patterns to try.
    patterns = [
        r"^```(?:[^\n]*)\s*\n(.*?)\s*```$",  # triple backticks with optional language tag
        r'^"""(?:[^\n]*)\s*\n(.*?)\s*"""$',  # triple double quotes with optional language tag
        r"^'''(?:[^\n]*)\s*\n(.*?)\s*'''$",  # triple single quotes with newline after the opening fence
        r'^"""(.*?)"""$',  # triple double quotes without newlines
        r"^'''(.*?)'''$",  # triple single quotes without newlines
    ]

    for pattern in patterns:
        regex = re.compile(pattern, re.DOTALL)
        m = regex.match(input_str)
        if m:
            return m.group(1).strip()
    return input_str


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
            elif isinstance(qa_data, list):
                # Process a list of QA dictionaries
                for item in qa_data:
                    if isinstance(item, dict):
                        qa_dict = {"question": item.get("question", None)}
                        thread_extracted["qa"].append(qa_dict)
    return thread_extracted


def extract_selected_threads(data, indexes=None):
    """
    Extracts data from the provided threads.

    Parameters:
      data (list): A list of threads loaded from JSON.
      indexes: Can be None, a single integer, a list of indices, or a tuple (start, end) for a contiguous slice.

    Returns:
      list: A list of dictionaries containing extracted thread details.
    """
    if indexes is None:
        return [extract_thread_details_fun(thread) for thread in data]
    elif isinstance(indexes, int):
        # Handle a single index by converting it to a list.
        if indexes < len(data):
            return [extract_thread_details_fun(data[indexes])]
        else:
            return []
    elif isinstance(indexes, list):
        return [extract_thread_details_fun(data[i]) for i in indexes if i < len(data)]
    elif isinstance(indexes, tuple):
        start, end = indexes
        if end > len(data):
            raise IndexError("Index out of range")
        return [extract_thread_details_fun(thread) for thread in data[start:end]]
    else:
        raise TypeError("indexes must be None, an int, a list, or a tuple.")


def extract_financial_data(input_data):
    """
    Extract financial data from the input.

    Accepted input formats:
    1. A JSON-formatted string representing a dictionary with the following structure:
         {
           "tool_input": {
             "file_path": "Path/to/your/file.json",
             "indexes": <optional_indexes_parameter>
           }
         }
       - file_path:
         - If this string ends with ".json", it is treated as a file path and loaded from disk.
         - Otherwise, it is treated as direct JSON content and processed as-is (ignoring indexes).
       - indexes: (Optional) When a valid file path is provided, this can be used to specify a subset
         of threads.

    2. A plain string representing either a file path (ending with .json) or a JSON-formatted string
       containing the data directly.

    Returns:
      A list of dictionaries representing the extracted financial information. On error, returns a dictionary
      with an "error" key. As a backup, if no proper extraction is possible but date patterns (YYYY-MM-DD) are found,
      they are returned in an "extracted_dates" field.
    """
    try:
        # If input_data is a string, strip any markdown code block markers.
        if isinstance(input_data, str):
            input_data = strip_code_fence(input_data)
            input_data = input_data.strip()
            # Replace common problematic smart quotes with standard double quotes.
            input_data = input_data.replace("“", '"').replace("”", '"')
            print("Raw input after stripping code fence:", repr(input_data))
            try:
                parsed = json.loads(input_data)
            except json.JSONDecodeError as je:
                print("JSON decoding failed:", je)
                # Fallback to ast.literal_eval if JSON decoding fails.
                try:
                    parsed = ast.literal_eval(input_data)
                except Exception as e:
                    raise ValueError(
                        "Parsing failed using both json.loads and ast.literal_eval: "
                        + str(e)
                    )
        else:
            parsed = input_data

        if isinstance(parsed, dict) and "tool_input" in parsed:
            tool_input = parsed["tool_input"]
            file_path = tool_input.get("file_path")
            indexes = tool_input.get("indexes", None)
            if not file_path:
                raise ValueError("The 'file_path' field is required in the tool_input.")
            if not isinstance(file_path, str):
                file_path = str(file_path)

            # If file_path ends with '.json', treat it as a file path.
            if file_path.strip().endswith(".json"):
                if os.path.exists(file_path):
                    data = open_json_file(file_path)
                else:
                    raise ValueError(f"File not found: {file_path}")
            else:
                # Otherwise, try to parse file_path as direct JSON content.
                try:
                    data = json.loads(file_path)
                except Exception:
                    raise ValueError(
                        "Provided file_path does not end with .json and is not valid JSON content."
                    )
                indexes = None  # Ignore indexes if direct JSON content is provided.
        else:
            data = parsed
            indexes = None

        financial_data = extract_selected_threads(data, indexes)
        return financial_data

    except Exception as e:
        # Backup: attempt to extract any date patterns (YYYY-MM-DD) from the input.
        pattern = r"\d{4}-\d{2}-\d{2}"
        extracted_dates = re.findall(
            pattern, input_data if isinstance(input_data, str) else str(input_data)
        )
        if extracted_dates:
            return {"extracted_dates": extracted_dates}
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
        name="extract_financial_informations",
        description="""
    Extracts structured financial information using the `extract_financial_data` function.

    **Trigger Condition:**
    - This tool is ONLY triggered when the input is a JSON string that represents a file path ending with ".json".
    - If the input is a JSON-formatted string containing data (i.e., the financial information is embedded directly), 
      this tool should NOT be triggered and the model will elaborate the string directly.

    **Accepted Input Formats:**

    **1. File Path Input (with indexes):**
    - The input must be a JSON string representing a dictionary with a `tool_input` key.
    - The `tool_input` value must be a dictionary with:
        - `file_path`: A string that **must end with .json**. This specifies the path to a JSON file on disk.
        - `indexes` (optional): 
            - If provided, it can be an integer (for a single thread), a list of indices, or a tuple (start, end)
              to extract a specific subset of threads from the file.
    - **Example:**
        '''{
            "tool_input": {
                "file_path": "data/financial_threads.json",
                "indexes": [3, 7, 15]
            }
        }'''

    **2. Direct JSON Content Input (Not Triggering This Tool):**
    - If the input is a JSON-formatted string representing the data directly 
      (i.e., not wrapped in a `tool_input` dictionary), this tool should NOT be triggered.
    - **Example:**
        '''{
            "pre_text": ["Example introduction"],
            "post_text": ["Example conclusion"],
            "table": [["Column1", "Column2"], ["Value1", "Value2"]],
            "qa": [{"question": "What is the key metric?"}]
        }'''

    **Output:**
    - The function returns a list of dictionaries, each containing:
        - "pre_text": A list of paragraphs that appear before the table.
        - "post_text": A list of paragraphs that appear after the table.
        - "table": A nested list representing tabular data.
        - "qa": A list of dictionaries, each containing at least a "question".
    - If an error occurs, a dictionary with an "error" key is returned.
    - As a backup, if no proper extraction is possible but date patterns (YYYY-MM-DD) are found in the input,
      they are returned in an "extracted_dates" field.
    """,
        func=extract_financial_data,
    ),
]
