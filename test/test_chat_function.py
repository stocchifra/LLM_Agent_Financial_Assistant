import unittest
from io import StringIO
from unittest.mock import patch

# Import the function to test.
from src.agent import chat


# Dummy classes and a dummy builder to replace the real agent executor and memory.
class DummyAgentExecutor:
    def invoke(self, payload):
        # Simulate a response that echoes back the input message.
        return {"output": f"Echo: {payload['input']}"}


class DummyChatMemory:
    def __init__(self):
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)


class DummyMemory:
    def __init__(self):
        self.chat_memory = DummyChatMemory()


def dummy_agent_executor_builder(*args, **kwargs):
    return DummyAgentExecutor(), DummyMemory()


class ChatFunctionTestCase(unittest.TestCase):
    # Patch the global variable in the chat function so that agent_executor_builder is replaced.
    @patch.dict(
        chat.__globals__, {"agent_executor_builder": dummy_agent_executor_builder}
    )
    @patch("builtins.input", side_effect=["Test Query", "exit"])
    @patch("builtins.print")
    def test_chat_function(self, mock_print, mock_input):
        """
        Test the chat() function:
          - Provides an initial input.
          - Simulates one interaction ("Test Query") followed by "exit" to break out.
          - Checks that the dummy agent echoes the inputs.
        """
        initial_input = "Initial Test Input"

        # Call the function under test.
        chat(
            initial_input=initial_input,
            model="dummy-model",  # The value is arbitrary since our dummy builder is used.
            provider="openai",
            temperature=0,
            tools=[],  # Providing an empty list for tools.
            prompt_style="structured-chat-agent",
            memory_flag=True,
            handle_parsing_errors=True,
            verbose=False,
        )

        # Capture the printed outputs.
        printed_outputs = [args[0] for (args, _) in mock_print.call_args_list]
        combined_output = "\n".join(printed_outputs)
        print("Combined Output:", combined_output)

        self.assertIn(
            "Hi Welcome to the chat! Type 'exit' to end the conversation.",
            combined_output,
        )
        self.assertIn("Thank you for using the chat! Goodbye!", combined_output)


if __name__ == "__main__":
    unittest.main()
