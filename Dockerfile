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
ARG RP_WORKSPACE=/workspace
ENV RP_WORKSPACE=$RP_WORKSPACE

# Override the default huggingface cache directory.
ENV HF_HOME="${RP_WORKSPACE}/.cache/huggingface/"

# Faster transfer of models from the hub to the container
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_XET_HIGH_PERFORMANCE=1

# Set up the working directory
ARG WORKSPACE_DIR=/app
ENV WORKSPACE_DIR=${WORKSPACE_DIR}
WORKDIR $WORKSPACE_DIR

# install git and wget
RUN apt-get update && apt-get install -y git wget

# Create virtualenv
ENV VIRTUAL_ENV=/app/venv/apps
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade packaging tools FIRST
RUN pip install --upgrade pip setuptools wheel

# Install base packages
RUN pip install --no-cache-dir \
    asyncio \
    requests \
    runpod

# Copy requirements
COPY requirements.txt .
COPY requirements--pre.txt .
COPY requirements--no-isolate.txt .

# Install normal requirements
RUN set -e; pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --pre -r requirements--pre.txt

# Install no-isolation requirements
RUN set -e; pip install --no-cache-dir --no-build-isolation -r requirements--no-isolate.txt

# install supersvg
RUN if [ ! -d "$WORKSPACE_DIR/supersvg" ]; then \
        set -e; \
        echo "================================"; \
        echo "installing supersvg"; \
        echo "================================"; \
        git clone https://github.com/am05mhz/SuperSVG.git supersvg; \
        cd "$WORKSPACE_DIR/supersvg"; \
        git submodule update --init --recursive; \
        cd diffvg; \
        python setup.py install; \
    else \
        echo "supersvg already exists, skipping clone."; \
    fi

# install layeredsvg
RUN if [ ! -d "$WORKSPACE_DIR/layeredsvg" ]; then \
        set -e; \
        echo "================================"; \
        echo "installing layeredsvg"; \
        echo "================================"; \
        git clone https://github.com/SZUVIZ/layered_vectorization.git layeredsvg; \
        cd "$WORKSPACE_DIR/layeredsvg"; \
        git submodule update --init --recursive; \
        cd diffvg; \
        python setup.py install; \
    else \
        echo "layeredsvg already exists, skipping clone."; \
    fi

RUN if [ ! -d "$WORKSPACE_DIR/layeredsvg/SAMRefiner" ]; then \
        set -e; \
        echo "cloning SAMRefiner"; \
        cd "$WORKSPACE_DIR/layeredsvg"; \
        git clone https://github.com/linyq2117/SAMRefiner.git; \
    fi

RUN set -e; if [ ! -d "$WORKSPACE_DIR/layeredsvg/LayeredVectorization/checkpoints" ]; then \
        echo "downloading checkpoints"; \
        cd "$WORKSPACE_DIR/layeredsvg"; \
        mkdir -p LayeredVectorization/checkpoints; \
        wget --tries=3 -O LayeredVectorization/checkpoints/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth; \
    fi

# install bezier splat
RUN if [ ! -d "$WORKSPACE_DIR/bezier" ]; then \
        set -e; \
        echo "================================"; \
        echo "installing bezier splat"; \
        echo "================================"; \
        git clone https://github.com/xiliu8006/Bezier_splatting.git bezier; \
        cd "$WORKSPACE_DIR/bezier"; \
        if [ ! -d "$WORKSPACE_DIR/bezier/gsplat" ]; then \
            git clone https://github.com/XingtongGe/gsplat.git; \
            cd "$WORKSPACE_DIR/bezier/gsplat"; \
            pip install --no-build-isolation .; \
        fi; \
    else \
        echo "bezier splat already exists, skipping clone."; \
    fi

# Copy all of our files into the container
COPY handler_runpod.py $WORKSPACE_DIR/handler_runpod.py
ADD sam3 $WORKSPACE_DIR/sam3
ADD bezier $WORKSPACE_DIR/bezier
ADD supersvg $WORKSPACE_DIR/supersvg
ADD layeredsvg $WORKSPACE_DIR/layeredsvg
ADD longcat $WORKSPACE_DIR/longcat
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