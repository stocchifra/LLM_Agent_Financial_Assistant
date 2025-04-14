import json
import os
import tempfile

from src.agent.agent_tools import (
    extract_financial_data,
    perform_math_calculus,
    strip_code_fence,
)


def test_strip_code_fence_basic_json():
    input_str = '```json\n{"key": "value"}\n```'
    result = strip_code_fence(input_str)
    assert result == '{"key": "value"}'


def test_strip_code_fence_no_fence():
    input_str = '{"key": "value"}'
    result = strip_code_fence(input_str)
    assert result == '{"key": "value"}'


def test_perform_math_calculus_valid():
    expression = "5 / 2"
    result = perform_math_calculus(expression)
    assert result == "2.5"


def test_perform_math_calculus_invalid():
    expression = "5 / 0"
    result = perform_math_calculus(expression)
    assert "Error" in result


def test_extract_financial_data_with_file_path():
    dummy_data = [
        {
            "pre_text": ["test pre"],
            "post_text": ["test post"],
            "table": [["a", "b"], [1, 2]],
            "qa": {"question": "How much?", "exe_ans": "2"},
        }
    ]
    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False, suffix=".json"
    ) as tmpfile:
        json.dump(dummy_data, tmpfile)
        tmpfile_path = tmpfile.name

    try:
        input_payload = json.dumps({"tool_input": {"file_path": tmpfile_path}})

        result = extract_financial_data(input_payload)
        assert isinstance(result, list)
        assert result[0]["qa"][0]["question"] == "How much?"
    finally:
        os.remove(tmpfile_path)


def test_extract_financial_data_direct_json():
    # Provide a JSON array (list) of threads
    data_str = json.dumps(
        [
            {
                "pre_text": ["Direct pre"],
                "post_text": ["Direct post"],
                "table": [["c", "d"], [3, 4]],
                "qa": {"question": "What is this?"},
            }
        ]
    )
    result = extract_financial_data(data_str)
    # Ensure the result is a list and the thread details are extracted correctly.
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["qa"][0]["question"] == "What is this?"
