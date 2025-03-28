FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive


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
ENV PYTHONUNBUFFERED=1 
RUN pip install uv

WORKDIR /app
#
## Copy the requirements file into the container
COPY requirements.eval.txt requirements.txt

RUN uv pip install --system -r requirements.txt --no-build-isolation
# Explicitly install to manage breaking changes
RUN uv pip install --system flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.5/
RUN uv pip install --system flash-attn --no-build-isolation

# Create empty directory and file for pyproject
RUN mkdir ./worker_api
RUN mkdir ./model_cache_dir

RUN touch ./worker_api/__init__.py
COPY scoring ./scoring
COPY utilities ./utilities
COPY constants ./constants
# Required for self installing module
COPY README.md .
COPY pyproject.toml .
COPY .git .git
RUN uv pip install --system -e .

COPY scoring/entrypoint.py .

ENTRYPOINT ["python", "entrypoint.py"]
