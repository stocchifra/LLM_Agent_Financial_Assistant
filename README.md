# Financial Assistant Agent Model

The goal of this project is to develop an LLM-powered agent capable of accurately answering financial questions by analyzing data, tables, and figures.

The model offers two main functionalities:

- **Chat mode:** Interactive conversation with integrated memory and context from previous messages.
- **Direct Answer mode:** Quick answers to questions without interactive chat.

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Features](#model-features)
- [Tools](#tools)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)
- [Roadmap](#roadmap)

---

## Installation

Clone the repository, add your API keys in the `.env` file, and run the following commands:

```bash
make docker-install  # Checks and installs Docker if necessary
make build           # Builds Docker image and installs dependencies via Poetry
make run             # Runs container interactively using environment variables from .env
```

---

## Usage

### Chat Mode

Interactively chat with the financial assistant. By default, it uses `gpt-4o` and `structured-chat-agent`. You can run the default chat using the command
```bash
make chat
```
prompt style, but you can select other models from OpenAI, Anthropic, or Google. Example of command:

```bash
docker run --rm -it --env-file .env financial_assistant_llm_agent --mode chat --model gpt-4o-mini --prompt_style react
```

### Direct Answer Mode

Get a fast response without interactive chat. Default settings use `gpt-4o` and the `react` prompt style. You can run the default direct answer using the command:
```bash
make DirectAnswer
```

Example command to change model and prompt:

```bash
docker run --rm -it --env-file .env financial_assistant_llm_agent --mode DirectAnswer --model gpt-4o-mini --prompt_style custom
```

---

## Model Features

The agent supports models from:

- **OpenAI**
- **Anthropic**
- **Google**

Supported prompt styles:

- **react**: Guides model reasoning through explicit thought and action steps.
- **fewshot-react**: Uses a few-shot learning strategy combined with react style for structured reasoning.
- **custom**: A JSON-chat style format, ideal for custom prompt engineering.
- **structured-chat-agent**: Optimized for structured conversational tasks.

---

## Tools

The agent includes two essential tools:

- **arithmetic\_calculator**: Performs arithmetic calculations from a given string. Useful since LLMs can struggle with complex mathematical expressions.
- **extract\_financial\_informations**: Extracts financial information from JSON files using specified indexes. It retrieves context (pre-text, post-text, tables) relevant to financial queries.

---

## Project Structure

```
├── data
├── docs
├── results
├── src
│   ├── agent
│   │   ├── __init__.py
│   │   ├── agent_builder.py
│   │   ├── agent_tools.py
│   │   ├── chat.py
│   │   ├── instant_answer.py
│   │   └── prompt_templates.py
│   ├── metrics
│   │   ├── __init__.py
│   │   ├── accuracy.py
│   │   ├── compute_metrics.py
│   │   └── llm_as_a_judge.py
│   ├── utils
│   │   ├── __init__.py
│   │   └── data_extractor.py
│   ├── demo.ipynb
│   ├── main.py
│   └── warnings_config.py
└── test
    ├── __init__.py
    ├── integration_test.py
    ├── test_agent_builder.py
    ├── test_agent_tools.py
    ├── test_chat_function.py
    ├── test_direct_answer.py
    ├── test_llm_as_a_judge.py
    ├── test_measure_accuracy.py
    ├── test_prompt_selector.py
    └── test_utils.py
```

---

## Running Tests
To run tests, make sure you're inside the Docker container and execute:
```bash
make test
```

## Contributing

Contributions are welcome! Please create an issue or submit a pull request.

---

## License

MIT License

---

## Author

Francesco Stocchi

---

## Roadmap

Next steps include:

- Further prompt engineering to enhance model performance.
- Integration of LangSmith for tracing model reasoning and tool invocation.
- Deployment into a cloud database for chat memory storage (e.g., Supabase, Firebase).
- Development of a user interface for easier user interactions.
