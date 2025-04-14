# Use the official Python 3.10 slim-bookworm image.
FROM python:3.10-slim-bookworm

# Set environment variables for Poetry and Python.
ENV POETRY_VERSION=1.5.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PYTHONUNBUFFERED=1

# Set the working directory.
WORKDIR /app
ENV PYTHONPATH=/app:/app/src

RUN rm -f /etc/apt/sources.list && \
    echo "deb http://ftp.us.debian.org/debian bookworm main contrib non-free" > /etc/apt/sources.list && \
    echo "deb http://ftp.us.debian.org/debian bookworm-updates main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian-security bookworm-security main contrib non-free" >> /etc/apt/sources.list && \
    apt-get update -o Acquire::ForceIPv4=true -o Acquire::Retries=3 && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        make && \
    rm -rf /var/lib/apt/lists/*


# Install Poetry and ensure it's globally available
RUN curl -sSL https://install.python-poetry.org | python3 - --version $POETRY_VERSION && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Copy dependency management files before installing deps
COPY pyproject.toml poetry.lock* ./

# Install project dependencies using Poetry
RUN poetry install --no-root

# Copy the rest of the project files into the container.
COPY . .


# (Optional) Expose a port if needed.
# EXPOSE 8000

# Set the default command to run the application.
ENTRYPOINT ["python", "-m", "src.main"]
