import unittest
from unittest.mock import patch

# Import the function to test.
from src.agent.instant_answer import direct_answer


# Dummy executor class that simulates a response with a comma-separated output.
class DummyAgentExecutor:
    def invoke(self, payload):
        # Return a dummy response with comma-separated values.
        return {"output": "Answer1,Answer2"}


# Dummy memory class (not used in the test but required by the builder interface).
class DummyMemory:
    pass


# Dummy builder function to replace the real agent_executor_builder.
def dummy_agent_executor_builder(*args, **kwargs):
    return DummyAgentExecutor(), DummyMemory()


class DirectAnswerTestCase(unittest.TestCase):
    # Patch the global variable in the direct_answer function's module so that agent_executor_builder
    # is replaced with our dummy implementation.
    @patch.dict(
        direct_answer.__globals__,
        {"agent_executor_builder": dummy_agent_executor_builder},
    )
    def test_direct_answer_returns_correct_answers(self):
        test_input = "What is the answer to life?"
        # Call the function under test.
        answers = direct_answer(input=test_input)

        # The dummy executor returns "Answer1,Answer2"
        # so we expect direct_answer to return a list with these two elements.
        self.assertEqual(
            answers,
            ["Answer1", "Answer2"],
            "The direct_answer function did not split the output correctly.",
        )


if __name__ == "__main__":
    unittest.main()
