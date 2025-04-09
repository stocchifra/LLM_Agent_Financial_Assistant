import pytest  # noqa: F401
from ConvFinQA_LLM_Project.data_handler import (
    extract_all_threads,
    extract_thread_details,
)


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
    assert len(result["qa"]) == 1
    assert result["qa"][0]["question"] == "What is 2+2?"
    assert result["qa"][0]["exe_ans"] == 4
    assert result["qa_1"][1]["question"] == "What is 3+3?"
    assert result["qa_1"][1]["exe_ans"] == 6


def test_extract_all_threads():
    dummy_threads = [
        {"pre_text": ["Text A"], "qa": {"question": "Q1", "exe_ans": 1}},
        {"pre_text": ["Text B"], "qa": {"question": "Q2", "exe_ans": 2}},
    ]
    results = extract_all_threads(dummy_threads)
    assert isinstance(results, list)
    assert len(results) == 2
    assert results[1]["qa"][0]["question"] == "Q2"
