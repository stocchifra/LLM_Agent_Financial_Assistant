import json


def open_json_file(file_path):
    """
    Open a JSON file and load its content.

    Parameters:
        file_path (str): Path to the JSON file.

    Returns:
        list: A list of dictionaries loaded from the JSON file.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def extract_thread_details(thread):
    """
    Extract required fields from a single thread.

    Parameters:
        thread (dict): A dictionary representing a JSON thread that may contain the keys:
                       'pre_text', 'post_text', 'table_ori', 'table', 'id', 'exe_ans',
                       and one or more QA fields such as 'qa', 'qa_0', 'qa_1'.

    Returns:
        dict: A dictionary with the following keys:
              - pre_text: list (or empty list if not present)
              - post_text: list (or empty list if not present)
              - table_ori: list (or empty list if not present)
              - table: list (or empty list if not present)
              - id: string (or empty string if not present)
              - exe_ans: extracted value from key 'exe_ans' at the thread level if present, otherwise None
              - qa: a list of QA dictionaries, each with:
                    * qa_field: the key name (e.g. "qa", "qa_0", "qa_1")
                    * question: the question string (or None)
                    * exe_ans: the exe_ans value corresponding to that qa (or None)
    """
    # Initialize the output dictionary with the defined keys.
    thread_extracted = {
        "pre_text": thread.get("pre_text", []),
        "post_text": thread.get("post_text", []),
        "table_ori": thread.get("table_ori", []),
        "table": thread.get("table", []),
        "id": thread.get("id", ""),
        "qa": [],  # will be a list of QA dictionaries
    }

    # Look for keys that are exactly "qa" or start with "qa_" (e.g., "qa_0", "qa_1", etc.)
    for key in thread:
        if key == "qa" or key.startswith("qa_"):
            qa_data = thread.get(key)
            if isinstance(qa_data, dict):
                qa_dict = {
                    "qa_field": key,
                    "question": qa_data.get("question", None),
                    "exe_ans": qa_data.get(
                        "exe_ans", None
                    ),  # Extract the exe_ans for each corresponding QA.
                }
                thread_extracted["qa"].append(qa_dict)

    return thread_extracted


def extract_selected_threads_processed(data, indexes=None):
    """
    Extrat data from a JSON file.
    Process a list of JSON threads and extract specified keys from each one.
    Parameters:
    data_path (str): Path to the JSON file containing thread data.
    Returns:
    list: A list of dictionaries with the extracted information.
    """
    if isinstance(data, str):
        data = open_json_file(data)

    if indexes is None:
        return [extract_thread_details(thread) for thread in data]
    elif isinstance(indexes, list):
        return [extract_thread_details(data[i]) for i in indexes if i < len(data)]
    elif isinstance(indexes, tuple):
        start, end = indexes
        if end > len(data):
            raise IndexError("Index out of range")
        return [extract_thread_details(thread) for thread in data[start:end]]


def get_exact_answers(single_sample):
    """
    Extracts and processes the 'exe_ans' (exact answer) values from all QA-related entries
    in a given sample dictionary. Looks for keys named 'qa' or starting with 'qa_'.

    Processing:
    - If the answer is a number, round it to 4 decimal places and convert to string.
    - If the answer is a string, convert to lowercase and remove all spaces.

    After extraction, the 'exe_ans' field is removed from the sample.

    Args:
        single_sample (dict): A dictionary containing QA data.

    Returns:
        tuple: A tuple containing:
            - list: A list of processed exact answers.
            - dict: The modified sample dictionary without the 'exe_ans' fields.
    """
    exact_answers = []
    for key in list(single_sample.keys()):
        if key == "qa" or key.startswith("qa_"):
            qa_data = single_sample.get(key)
            # Handle the case where qa_data is a list of dictionaries
            if isinstance(qa_data, list):
                for item in qa_data:
                    if isinstance(item, dict) and "exe_ans" in item:
                        answer = item["exe_ans"]
                        if isinstance(answer, (int, float)):
                            processed = str(round(answer, 4))
                        elif isinstance(answer, str):
                            processed = answer.lower().replace(" ", "")
                        else:
                            processed = str(answer) if answer is not None else None
                        exact_answers.append(processed)
                        del item["exe_ans"]
            # Handle the case where qa_data is a dictionary
            elif isinstance(qa_data, dict) and "exe_ans" in qa_data:
                answer = qa_data["exe_ans"]
                if isinstance(answer, (int, float)):
                    processed = str(round(answer, 4))
                elif isinstance(answer, str):
                    processed = answer.lower().replace(" ", "")
                else:
                    processed = str(answer) if answer is not None else None
                exact_answers.append(processed)
                del qa_data["exe_ans"]
    return exact_answers, single_sample
