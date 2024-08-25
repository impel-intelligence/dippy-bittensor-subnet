FROM python:3.12

RUN python -m pip install --upgrade pip

# env variable required by uv
ENV CONDA_PREFIX=/opt/conda
RUN pip install uv

WORKDIR /app
#
## Copy the requirements file into the container
COPY requirements_val_api.txt requirements.txt

RUN uv pip install --system -r requirements.txt --no-build-isolation
# Create empty directory and file for pyproject
RUN mkdir ./dippy_validation_api
RUN touch ./dippy_validation_api/__init__.py
COPY scoring ./scoring
COPY utilities ./utilities
COPY template ./template
COPY model ./model
COPY constants ./constants
# Required for self installing module
COPY README.md .
COPY pyproject.toml .
COPY .git .git
RUN uv pip install -e .

COPY dippy_validation_api/validation_api.py .

ENTRYPOINT ["python", "validation_api.py"]