import json


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


def extract_all_threads(data, indexes=None):
    """
    Process a list of JSON threads and extract specified keys from each one.
    Parameters:
    data (list): List of thread dictionaries loaded from a JSON file.
    Returns:
    list: A list of dictionaries with the extracted information.
    """
    if indexes is None:
        return [extract_thread_details(thread) for thread in data]
    elif isinstance(indexes, list):
        return [extract_thread_details(data[i]) for i in indexes if i < len(data)]
    elif isinstance(indexes, tuple):
        start, end = indexes
        if end > len(data):
            raise IndexError("Index out of range")
        return [extract_thread_details(thread) for thread in data[start:end]]


# Example usage:
if __name__ == "__main__":
    # Replace the path below with your actual JSON file path.
    data_path = "/Users/francescostocchi/ConvFinQA_LLM_Project/data/train.json"

    # Open the JSON file and load the data.
    with open(data_path, "r") as f:
        data = json.load(f)

    # Extract details from all threads.
    extracted_threads = extract_all_threads(data, indexes=(0, 2))

    # Optionally, print the first extracted thread for inspection.
    print(extracted_threads[2]["pre_text"])
