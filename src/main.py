#!/usr/bin/env python3
import argparse
import warnings

from langsmith.utils import LangSmithMissingAPIKeyWarning

from src import agent_executor_builder  # Adjust path if needed
from src import tools  # Adjust path if needed

warnings.filterwarnings("ignore", category=LangSmithMissingAPIKeyWarning)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*agent.agent_builder.*"
)
# Import the two functionalities:
# Assuming that chat functionality and direct answer (Q&A) functionality have been implemented as specified.
from src import chat  # This module contains the interactive chat function
from src import direct_answer  # This module contains the direct answer (Q&A) function

# from src import warnings_config  # noqa: F401


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate the agent functionalities: chat or DirectAnswer (Q&A)."
    )
    parser.add_argument(
        "--mode",
        choices=["chat", "DirectAnswer"],
        required=True,
        help="Mode to run: 'chat' for interactive chat or 'DirectAnswer' for Q&A.",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="The input for the agent. In chat mode, it can be an initial message; in Q&A mode, the question/prompt.",
    )

    # Additional parameters that the user can choose for both modes:
    parser.add_argument(
        "--model", type=str, default="gpt-4o", help="Model to use (default: gpt-4o)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        help="Provider to use (default: openai)",
    )
    # parser.add_argument("--temperature", type=float, default=0, help="Temperature setting (default: 0)")

    # Additional parameters for Q&A mode (only used in DirectAnswer mode):
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="react",
        help="Prompt style (default: react)",
    )
    parser.add_argument(
        "--memory_flag",
        action="store_true",
        help="Enable memory (only for DirectAnswer mode)",
    )
    parser.add_argument(
        "--handle_parsing_errors",
        action="store_true",
        help="Handle parsing errors (only for DirectAnswer mode)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (only for DirectAnswer mode)",
    )

    args = parser.parse_args()

    # If --input wasn't provided, prompt the user interactively
    if not args.input:
        args.input = input("Please enter your input: ")

    if args.mode == "chat":
        # For chat mode, the following parameters remain fixed:\
        temperature = 0
        fixed_prompt_style = "structured-chat-agent"
        fixed_memory_flag = True
        fixed_handle_parsing_errors = True
        fixed_verbose = False

        # print("Starting interactive chat mode. Type 'exit' to end the conversation.\n")
        # Call the chat function with the parameters for model, provider, and temperature coming from args.
        chat(
            initial_input=args.input,
            model=args.model,  # User-provided model
            provider=args.provider,  # User-provided provider
            temperature=temperature,  # User-provided temperature
            tools=tools,
            prompt_style=fixed_prompt_style,  # Fixed for chat mode
            memory_flag=fixed_memory_flag,  # Fixed for chat mode
            handle_parsing_errors=fixed_handle_parsing_errors,  # Fixed for chat mode
            verbose=fixed_verbose,  # Fixed for chat mode
        )

    elif args.mode == "DirectAnswer":
        # the following parameters remain fixed:\
        temperature = 0
        # In Q&A mode, let the user change all parameters.
        print("Running Q&A functionality with the following parameters:")
        print(f"Model: {args.model}")
        print(f"Provider: {args.provider}")
        print(f"Temperature: {temperature}")
        print(f"Prompt style: {args.prompt_style}")
        print(f"Memory enabled: {args.memory_flag}")
        print(f"Handle parsing errors: {args.handle_parsing_errors}")
        print(f"Verbose output: {args.verbose}\n")

        # Call the direct_answer function with user-specified parameters.
        answers = direct_answer(
            input=args.input,
            model=args.model,
            provider=args.provider,
            temperature=temperature,
            tools=tools,
            prompt_style=args.prompt_style,
            memory_flag=args.memory_flag,
            handle_parsing_errors=args.handle_parsing_errors,
            verbose=args.verbose,
        )
        # Print answers (assuming answers is a list of strings)
        print("Answers:")
        for answer in answers:
            print(answer.strip())


if __name__ == "__main__":
    main()
