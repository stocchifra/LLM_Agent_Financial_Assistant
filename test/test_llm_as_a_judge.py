import json
import unittest
from unittest.mock import MagicMock, patch

from src.metrics.llm_as_a_judge import evaluate_answer


class TestEvaluateAnswer(unittest.TestCase):
    @patch("langchain_openai.ChatOpenAI")
    def test_evaluate_answer_correct(self, MockChatOpenAI):
        """
        Test evaluate_answer with a dummy correct response.
        """
        # Create a dummy response message with a JSON string output.
        dummy_response = MagicMock()
        dummy_response.content = json.dumps(
            {
                "score": 1,
                "explanation": "The candidate number is within 1% of the expected value.",
            }
        )

        # Set up the mock to return our dummy response
        instance = MockChatOpenAI.return_value
        instance.predict_messages.return_value = dummy_response

        # Define dummy input values
        question = "What is 2 + 2?"
        expected_answer = "4"
        candidate_answer = "The sum of 2 + 2 is 4."

        # Call the evaluate_answer function
        result = evaluate_answer(question, expected_answer, candidate_answer)

        # Validate the output
        self.assertEqual(result["score"], 1)

    @patch("langchain_openai.ChatOpenAI")
    def test_evaluate_answer_incorrect(self, MockChatOpenAI):
        """
        Test evaluate_answer with a dummy incorrect response.
        """
        # Create a dummy response message with a JSON string output.
        dummy_response = MagicMock()
        dummy_response.content = json.dumps(
            {
                "score": 0,
                "explanation": "The candidate answer does not match the expected yes/no response.",
            }
        )

        # Set up the mock to return our dummy response
        instance = MockChatOpenAI.return_value
        instance.predict_messages.return_value = dummy_response

        # Define dummy input values for a yes/no question
        question = "Is the sky blue?"
        expected_answer = "yes"
        candidate_answer = "No"

        # Call the evaluate_answer function
        result = evaluate_answer(question, expected_answer, candidate_answer)

        # Validate the output
        self.assertEqual(result["score"], 0)


if __name__ == "__main__":
    unittest.main()
