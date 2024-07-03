
## Set environment variables
#ENV PYTHONUNBUFFERED=1 \
#    DEBIAN_FRONTEND=noninteractive \

FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

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
COPY requirements.eval.txt .

RUN uv pip install --system -r requirements.eval.txt --no-build-isolation

COPY scoring ./scoring
COPY utilities ./utilities
COPY template ./template
COPY model ./model
COPY constants ./constants
# Required for self installing module
COPY requirements.eval.txt requirements.txt
COPY README.md .
COPY setup.py .
RUN uv pip install -e .

COPY scoring/entrypoint.py .

ENTRYPOINT ["python", "entrypoint.py"]
