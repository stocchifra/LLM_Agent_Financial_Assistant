import json
import os
import random
import tempfile

import pytest

# Import the module where the function is defined so we can monkeypatch correctly
import src.metrics.accuracy as accuracy_module

# Tools used in the agent_executor_builder
from src.agent.agent_builder import tools


class DummyAgentExecutor:
    def invoke(self, input_data):
        # Simulate output "1.0" (numeric) as a single answer
        return {"output": "1.0"}


def dummy_agent_executor_builder(*args, **kwargs):
    # Simulate the return of the agent_executor and no memory
    return DummyAgentExecutor(), None


def create_dummy_data_file():
    # Dummy threads all with the same expected answer (1.0)
    dummy_data = [
        {
            "pre_text": ["Sample pre 1"],
            "post_text": ["Sample post 1"],
            "table": [["Col1", "Col2"], ["A", "B"]],
            "qa": {"question": "What is test?", "exe_ans": 1.0},
        },
        {
            "pre_text": ["Sample pre 2"],
            "post_text": ["Sample post 2"],
            "table": [["Col1", "Col2"], ["C", "D"]],
            "qa": {"question": "What is test?", "exe_ans": 1.0},
        },
        {
            "pre_text": ["Sample pre 3"],
            "post_text": ["Sample post 3"],
            "table": [["Col1", "Col2"], ["E", "F"]],
            "qa": {"question": "What is test?", "exe_ans": 1.0},
        },
    ]
    tmpfile = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json")
    json.dump(dummy_data, tmpfile)
    tmpfile_path = tmpfile.name
    tmpfile.close()
    return tmpfile_path


def test_measure_accuracy_metric_calculations(monkeypatch):
    # Patch exactly where it's used: inside src.metrics.accuracy
    monkeypatch.setattr(
        accuracy_module, "agent_executor_builder", dummy_agent_executor_builder
    )

    # Fix the random seed for reproducibility
    random.seed(42)

    # Generate test data
    dummy_file_path = create_dummy_data_file()

    # Run the function under test
    metrics = accuracy_module.measure_accuracy(
        data_path=dummy_file_path,
        model="dummy-model",
        provider="dummy-provider",
        prompt_style="react",
        tolerance=0.005,
        number_samples=3,
        tools=tools,
        temperature=0,
        memory_flag=False,
    )

    # Cleanup
    os.remove(dummy_file_path)

    # Assert expected outputs
    assert all(score == 1 for score in metrics["accuracy_measurements"])
    assert metrics["mean_accuracy"] == 1.0
    assert metrics["mae"] == 0.0
    assert metrics["mse"] == 0.0
