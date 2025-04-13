import json
import os
import tempfile

import pytest  # noqa: F401

from src import (
    extract_selected_threads,
    extract_thread_details,
    open_json_file,
)


def test_open_json_file():
    # Create dummy data that matches the expected structure
    dummy_data = [
        {
            "pre_text": ["Some context before the table."],
            "post_text": ["Some context after the table."],
            "table": [["Header1", "Header2"], ["Row1Val1", "Row1Val2"]],
            "qa": {"question": "What is the value of Header1?", "exe_ans": "Row1Val1"},
        }
    ]

    # Write it to a temporary JSON file
    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False, suffix=".json"
    ) as tmpfile:
        json.dump(dummy_data, tmpfile)
        tmpfile_path = tmpfile.name

    try:
        # Call your function with the dummy file path
        data = open_json_file(tmpfile_path)

        # Assertions
        assert isinstance(data, list)
        assert len(data) > 0
        assert isinstance(data[0], dict)
        assert "pre_text" in data[0]
        assert "post_text" in data[0]
        assert "table" in data[0]
        assert "qa" in data[0]
    finally:
        os.remove(tmpfile_path)


def test_extract_thread_details():
    dummy_thread = {
        "pre_text": ["Example pre text"],
        "post_text": ["Example post text"],
        "table": [["header1", "header2"], ["val1", "val2"]],
        "id": "test_id",
        "qa": {"question": "What is 2+2?", "exe_ans": 4},
        "qa_1": {"question": "What is 3+3?", "exe_ans": 6},
    }

    result = extract_thread_details(dummy_thread)

    assert result["pre_text"] == ["Example pre text"]
    assert result["post_text"] == ["Example post text"]
    assert result["id"] == "test_id"

    assert isinstance(result["qa"], list)
    assert len(result["qa"]) == 2

    # Check the QA contents (order might vary if using dicts)
    qa_fields = {qa["qa_field"]: qa for qa in result["qa"]}

    assert qa_fields["qa"]["question"] == "What is 2+2?"
    assert qa_fields["qa"]["exe_ans"] == 4

    assert qa_fields["qa_1"]["question"] == "What is 3+3?"
    assert qa_fields["qa_1"]["exe_ans"] == 6


def test_extract_all_threads():
    dummy_threads = [
        {"pre_text": ["Text A"], "qa": {"question": "Q1", "exe_ans": 1}},
        {"pre_text": ["Text B"], "qa": {"question": "Q2", "exe_ans": 2}},
    ]

    # Create a temporary JSON file
    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False, suffix=".json"
    ) as tmpfile:
        json.dump(dummy_threads, tmpfile)
        tmpfile_path = tmpfile.name

    try:
        # Call the function with the path
        results = extract_selected_threads(tmpfile_path)

        # Assertions
        assert isinstance(results, list)
        assert len(results) == 2
        assert results[1]["qa"][0]["question"] == "Q2"
    finally:
        # Clean up the temporary file
        os.remove(tmpfile_path)
