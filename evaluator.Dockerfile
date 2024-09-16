
## Set environment variables
#ENV PYTHONUNBUFFERED=1 \
#    DEBIAN_FRONTEND=noninteractive \

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# build-essential: installs gcc which is needed to install some deps like rasterio
# libGL1: needed to avoid following error when using cv2
# ImportError: libGL.so.1: cannot open shared object file: No such file or directory
# See https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt-get update && \
    apt-get install -y wget=1.* build-essential libgl1 curl git tree && \
    apt-get autoremove && apt-get autoclean && apt-get clean

RUN python -m pip install --upgrade pip

# env variable required by uv
ENV CONDA_PREFIX=/opt/conda
RUN pip install uv

WORKDIR /app
#
## Copy the requirements file into the container
COPY requirements.eval.txt requirements.txt

RUN uv pip install --system -r requirements.txt --no-build-isolation
# Create empty directory and file for pyproject
RUN mkdir ./dippy_validation_api
RUN mkdir ./model_cache_dir
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
RUN uv pip install --system -e .

COPY scoring/entrypoint.py .

ENTRYPOINT ["python", "entrypoint.py"]
