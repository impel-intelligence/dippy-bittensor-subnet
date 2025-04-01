FROM python:3.12
COPY --from=ghcr.io/astral-sh/uv:0.4 /uv /bin/uv
ENV UV_SYSTEM_PYTHON=1
WORKDIR /app

## Copy the requirements file into the container
COPY requirements.validator.txt requirements.txt

RUN uv pip install -r requirements.txt --prerelease=allow

# Create empty directory and file for pyproject
RUN mkdir ./worker_api
RUN touch ./worker_api/__init__.py
COPY scoring ./scoring
COPY utilities ./utilities
COPY constants ./constants
COPY common ./common

# COPY .env .
COPY README.md .
COPY pyproject.toml .
COPY .git .git
RUN uv pip install -e .

COPY neurons/model_queue.py .

ENTRYPOINT ["python", "model_queue.py"]


