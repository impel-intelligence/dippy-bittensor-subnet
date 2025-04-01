FROM python:3.12
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /bin/uv
# RUN uv venv /opt/venv
# # Use the virtual environment automatically
# ENV VIRTUAL_ENV=/opt/venv
# # Place entry points in the environment at the front of the path
# ENV PATH="/opt/venv/bin:$PATH"
ENV UV_SYSTEM_PYTHON=1
WORKDIR /app
#
# Install git and git-lfs
RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*

# Initialize git-lfs
RUN git lfs install

## Copy the requirements file into the container
COPY requirements.api.txt requirements.txt

RUN uv pip install -r requirements.txt

COPY scoring ./scoring
COPY worker_api ./worker_api
COPY utilities ./utilities
COPY common ./common
COPY constants ./constants
# Required for self installing module
COPY README.md .
COPY pyproject.toml .
COPY .git .git
RUN uv pip install -e .

COPY worker_api/validation_api.py .

CMD ["python", "validation_api.py"]