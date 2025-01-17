FROM python:3.12-slim

# Set environment variables to prevent Python from writing .pyc files and for unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Poetry
RUN apt-get update && \
    apt-get install --no-install-suggests --no-install-recommends --yes pipx
ENV PATH="/root/.local/bin:${PATH}"
RUN pipx install poetry

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-interaction --no-ansi

# Copy the entire project into the container
COPY . .

CMD ["poetry", "run", "torchrun", "--nnodes", "1", "--nproc_per_node", "8", "fsdp_playground.py"]