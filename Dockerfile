# Use an official Runpod base image
FROM runpod/pytorch:1.0.2-cu1281-torch271-ubuntu2204

# Set the shell and enable pipefail for better error handling
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Environment variables
ENV PYTHONUNBUFFERED=1 
ENV SHELL=/bin/bash 
ENV DEBIAN_FRONTEND=noninteractive

# Set basic environment variables
ARG PYTHON_VERSION
ARG TORCH_VERSION
ARG CUDA_VERSION

ENV ARG_PYTHON_VERSION=${PYTHON_VERSION}
ENV ARG_TORCH_VERSION=${TORCH_VERSION}
ENV ARG_CUDA_VERSION=${CUDA_VERSION}

# Supported modes: pod, serverless
ARG MODE_TO_RUN=pod
ENV MODE_TO_RUN=$MODE_TO_RUN

# set envs
ARG VLLM_PORT=8080
ENV VLLM_PORT=$VLLM_PORT
ARG SAM3_PORT=8080
ENV SAM3_PORT=$SAM3_PORT
ARG USE_VLLM=0
ENV USE_VLLM=$USE_VLLM
ARG SUPERSVG_PORT=8070
ENV SUPERSVG_PORT=$SUPERSVG_PORT
ARG BEZIER_PORT=8060
ENV BEZIER_PORT=$BEZIER_PORT
ARG COMBINED_PORT=8060
ENV COMBINED_PORT=$COMBINED_PORT
ARG START_SUPERSVG=0
ENV START_SUPERSVG=$START_SUPERSVG
ARG START_BEZIER=0
ENV START_BEZIER=$START_BEZIER
ARG START_COMBINED=0
ENV START_COMBINED=$START_COMBINED
ARG SSH_KEY=0
ENV SSH_KEY=$SSH_KEY

ARG HF_TOKEN=0
ENV HF_TOKEN=$HF_TOKEN


# Set the default workspace directory
ENV RP_WORKSPACE=/workspace

# Override the default huggingface cache directory.
ENV HF_HOME="${RP_WORKSPACE}/.cache/huggingface/"

# Faster transfer of models from the hub to the container
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_XET_HIGH_PERFORMANCE=1

# Set up the working directory
ARG WORKSPACE_DIR=/app
ENV WORKSPACE_DIR=${WORKSPACE_DIR}
WORKDIR $WORKSPACE_DIR

# Install dependencies in a single RUN command to reduce layers and clean up in the same layer to reduce image size

RUN add-apt-repository --yes ppa:deadsnakes/ppa && \
    apt-get update --yes --quiet && \
    apt-get install --yes --quiet --no-install-recommends \
    software-properties-common \
    gpg-agent \
    build-essential \
    apt-utils \
    ca-certificates \
    curl \
    git

# Create and activate a Python virtual environment
RUN python3 -m venv /app/venv/apps
RUN python3 -m venv /app/venv/bezier

# Install Python packages
RUN source /app/venv/apps/bin/activate
RUN pip install --no-cache-dir \
    asyncio \
    requests \
    runpod

# Install requirements.txt
COPY requirements.txt ./
COPY requirements--pre.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "setuptools<82.0.0" && \
    pip install --no-cache-dir --upgrade huggingface_hub && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-build-isolation -r requirements--no-isolate.txt && \
    pip install --no-cache-dir --pre -r requirements--pre.txt

# RUN git clone https://github.com/facebookresearch/sam3.git && \
#     cd sam3 && \
#     pip install -e .    

# Copy all of our files into the container
# COPY handler.py $WORKSPACE_DIR/handler.py
ADD sam3 $WORKSPACE_DIR/sam3
ADD bezier $WORKSPACE_DIR/bezier
ADD supersvg $WORKSPACE_DIR/supersvg
ADD layeredsvg $WORKSPACE_DIR/layeredsvg
ADD templates $WORKSPACE_DIR/templates
COPY combined_api.py $WORKSPACE_DIR/combined_api.py
COPY common_utils.py $WORKSPACE_DIR/common_utils.py
COPY start.sh /start.sh
COPY logo.txt /etc/runpod.txt

# Make sure start.sh is executable
RUN chmod +x /start.sh

# Make sure that the start.sh is in the path
RUN ls -la /start.sh

CMD /start.sh