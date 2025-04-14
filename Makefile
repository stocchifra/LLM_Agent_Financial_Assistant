.PHONY: build run

# Build the Docker image with the tag 'financial_assistant_llm_agent'
build:
	docker build -t financial_assistant_llm_agent .

# Run the container interactively, loading environment variables from the .env file
run:
	docker run --rm -it --env-file .env financial_assistant_llm_agent

# Install Docker if not installed (Linux only)
docker-install:
	@if ! command -v docker >/dev/null 2>&1; then \
		echo "Docker not found. Attempting to install Docker (Linux only)..."; \
		if [ "$$(uname)" = "Linux" ]; then \
			sudo apt-get update && sudo apt-get install -y docker.io; \
			sudo systemctl enable --now docker; \
			sudo usermod -aG docker $$(whoami); \
			echo "Docker installed. You may need to log out and log back in."; \
		else \
			echo "🚫 Automatic install is only supported on Linux."; \
			echo "🔧 Please install Docker manually from https://www.docker.com/products/docker-desktop"; \
		fi \
	else \
		echo "✅ Docker is already installed."; \
	fi

# run the container with the chat mode
chat:
	docker run --rm -it --env-file .env financial_assistant_llm_agent --mode chat
# run the container with the direct answer mode
direct_question:
	docker run --rm -it --env-file .env financial_assistant_llm_agent --mode DirectAnswer --verbose --handle_parsing_errors

# run the test using pytest
unit_test:
	docker run --rm -it --env-file .env --entrypoint pytest financial_assistant_llm_agent test
