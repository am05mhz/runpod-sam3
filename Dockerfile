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

# set vllm port
ARG VLLM_PORT=8080
ENV VLLM_PORT=$VLLM_PORT

# Set the default workspace directory
ENV RP_WORKSPACE=/workspace

# Override the default huggingface cache directory.
ENV HF_HOME="${RP_WORKSPACE}/.cache/huggingface/"

# Faster transfer of models from the hub to the container
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_XET_HIGH_PERFORMANCE=1

# Set up the working directory
ARG WORKSPACE_DIR=/workspace/app
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
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install Python packages
RUN pip install --no-cache-dir \
    asyncio \
    requests \
    runpod

# Install requirements.txt
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade huggingface_hub && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install vllm --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128

RUN git clone https://github.com/facebookresearch/sam3.git && \
    cd sam3 && \
    pip install -e .    

# Delete the default start.sh file from Runpod (so we can replace it with our own below)
RUN rm ../start.sh

# Copy all of our files into the container
# COPY handler.py $WORKSPACE_DIR/handler.py
COPY sam3/server.py $WORKSPACE_DIR/sam3/server.py
COPY start.sh $WORKSPACE_DIR/start.sh

# Make sure start.sh is executable
RUN chmod +x start.sh

# Make sure that the start.sh is in the path
RUN ls -la $WORKSPACE_DIR/start.sh

CMD $WORKSPACE_DIR/start.sh