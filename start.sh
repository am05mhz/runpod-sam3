#!/bin/bash
set -e  # Exit the script if any statement returns a non-true return value

# Set workspace directory from env or default
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"

# Start nginx service
start_nginx() {
    echo "Starting Nginx service..."
    service nginx start
}

# # Execute script if exists
# execute_script() {
#     local script_path=$1
#     local script_msg=$2
#     if [[ -f ${script_path} ]]; then
#         echo "${script_msg}"
#         bash ${script_path}
#     fi
# }

# Setup ssh
setup_ssh() {
    if [[ $SSH_KEY ]]; then
        echo ""
        echo "========================================================"
        echo "Setting up SSH..."
        echo "========================================================"
        mkdir -p ~/.ssh
        echo "$SSH_KEY" >> ~/.ssh/authorized_keys
        chmod 700 -R ~/.ssh
        # Generate SSH host keys if not present
        generate_ssh_keys
        service ssh start
        echo "SSH host keys:"
        cat /etc/ssh/*.pub
    else
        echo "SSH public key not setup"
    fi
}

setup_sam3() {
    echo ""
    echo "========================================================"
    echo "installing sam3..."
    echo "========================================================"
    source /app/venv/apps/bin/activate
    if [ ! -d "/workspace/apps" ]; then
        mkdir -p /workspace/apps
    fi
    cp -r /app/sam3 /workspace/apps/
    cp /app/requirements*.txt /workspace/apps/sam3
    cd /workspace/apps/sam3
    pip install --upgrade --no-cache-dir -r requirements.txt
    pip install --upgrade --pre --no-cache-dir -r requirements--pre.txt
}

setup_supersvg() {
    echo ""
    echo "========================================================"
    echo "installing supersvg..."
    echo "========================================================"
    source /app/venv/apps/bin/activate
    if [ ! -d "/workspace/apps" ]; then
        mkdir -p /workspace/apps
    fi
    cd /workspace/apps
    if [ ! -d "/workspace/apps/supersvg" ]; then
        git clone https://github.com/sjtuplayer/SuperSVG.git supersvg
    fi
    if [ ! -d "/workspace/apps/supersvg/DiffVG" ]; then
        rm -r /workspace/apps/supersvg/DiffVG
    fi
    if [ ! -d "diffvg" ]; then
        git clone https://github.com/BachiLi/diffvg.git
    fi
    cp -r /app/supersvg /workspace/apps/
    cd supersvg
    pip install --upgrade --no-cache-dir -r requirements.txt
    cd diffvg
    git submodule update --init --recursive
    pip install --no-build-isolation .
}

setup_combined() {
    echo ""
    echo "========================================================"
    echo "installing combined api..."
    echo "========================================================"
    source /app/venv/apps/bin/activate
    cp /app/combined_api.py /workspace/apps/
    cp /app/common_utils.py /workspace/apps/
    cp -r /app/templates /workspace/apps/
}

setup_layeredsvg() {
    echo ""
    echo "========================================================"
    echo "installing layeredsvg..."
    echo "========================================================"
    source /app/venv/apps/bin/activate
    if [ ! -d "/workspace/apps" ]; then
        mkdir -p /workspace/apps
    fi
    cd /workspace/apps
    if [ ! -d "/workspace/apps/layeredsvg" ]; then
        git clone https://github.com/SZUVIZ/layered_vectorization.git layeredsvg
    fi
    cd /workspace/apps/layeredsvg
    if [ ! -d "SAMRefiner" ]; then
        git clone https://github.com/linyq2117/SAMRefiner.git
    fi
    if [ ! -d "diffvg" ]; then
        git clone https://github.com/BachiLi/diffvg.git
    fi
    cd diffvg
    git submodule update --init --recursive
    cp -r /app/layeredsvg /workspace/apps/
    pip install --no-build-isolation .
    cd /workspace/apps/layeredsvg
    pip install --upgrade --no-cache-dir -r requirements.txt
    mkdir -p LayeredVectorization/checkpoints
    cd LayeredVectorization/checkpoints
    if [ ! -f "sam_vit_h_4b8939.pth" ]; then
        wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    fi
}

setup_longcat() {
    echo ""
    echo "========================================================"
    echo "installing longcat..."
    echo "========================================================"
    source /app/venv/apps/bin/activate
    if [ ! -d "/workspace/apps" ]; then
        mkdir -p /workspace/apps
    fi
    cp -r /app/longcat /workspace/apps/
    cd /workspace/apps/longcat
}

setup_bezier() {
    echo ""
    echo "========================================================"
    echo "installing bezier splatting..."
    echo "========================================================"
    source /app/venv/apps/bin/activate
    if [ ! -d "/workspace/apps" ]; then
        mkdir -p /workspace/apps
    fi
    cd /workspace/apps
    if [ ! -d "/workspace/apps/bezier" ]; then
        git clone https://github.com/xiliu8006/Bezier_splatting.git bezier
    fi
    cp -r /app/bezier /workspace/apps/
    cd /workspace/apps/bezier
    pip install --upgrade --no-cache-dir -r requirements.txt
    if [ ! -d "/workspace/apps/bezier/gsplat" ] || [ ! -f "/workspace/apps/bezier/gsplat/setup.py" ]; then
        if [ -d "/workspace/apps/bezier/gsplat" ]; then
            rm -rf /workspace/apps/besier/gsplat
        fi
        git clone https://github.com/XingtongGe/gsplat.git
    fi
    cd /workspace/apps/bezier/gsplat
    ls -la
    pip install -e .
}

start_sam3() {
    echo ""
    echo "========================================================"
    echo "starting sam3..."
    echo "========================================================"
    source /app/venv/apps/bin/activate
    cd /workspace/apps/sam3
    nohup python server.py --port=$SAM3_PORT > /proc/self/fd/1 2>&1 &
}

start_supersvg() {
    if [[ $START_SUPERSVG ]]; then
        echo ""
        echo "========================================================"
        echo "starting supersvg..."
        echo "========================================================"
        source /app/venv/apps/bin/activate
        cd /workspace/apps/supersvg
        nohup python server.py --port=$SUPERSVG_PORT > /proc/self/fd/1 2>&1 &
    else
        echo "not starting supersvg"
    fi
}

start_combined() {
    if [[ $START_COMBINED ]]; then
        echo ""
        echo "========================================================"
        echo "starting combined api..."
        echo "========================================================"
        source /app/venv/apps/bin/activate
        cd /workspace/apps
        nohup python combined_api.py --port=$COMBINED_PORT > /proc/self/fd/1 2>&1 &
    else
        echo "not starting combined api"
    fi
}

start_bezier() {
    if [[ $START_BEZIER ]]; then
        echo ""
        echo "========================================================"
        echo "starting bezier splatting..."
        source /app/venv/apps/bin/activate
        cd /workspace/apps/bezier
        nohup python server.py --port=$BEZIER_PORT > /proc/self/fd/1 2>&1 &
    else
        echo "not starting bezier splat"
    fi
}

setup_vllm() {
    echo ""
    echo "========================================================"
    echo "setup vllm..."
    if [[ $USE_VLLM ]]; then
        pip install vllm --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128
        echo "starting vllm..."
        vllm serve Qwen/Qwen3-VL-8B-Thinking --max-num-seqs 2 --tensor-parallel-size 1 --gpu-memory-utilization 0.55 --allowed-local-media-path / --enforce-eager --port $VLLM_PORT
    else
        echo "not setup vllm"
    fi
}

# Generate SSH host keys
generate_ssh_keys() {
    ssh-keygen -A
}

# Export env vars
export_env_vars() {
    echo "Exporting environment variables..."
    printenv | grep -E '^RUNPOD_|^PATH=|^_=' | awk -F = '{ print "export " $1 "=\"" $2 "\"" }' >> /etc/rp_environment
    echo 'source /etc/rp_environment' >> ~/.bashrc
}

# Call Python handler if mode is serverless or both
call_python_handler() {
    echo "Calling Python handler.py..."
    python $WORKSPACE_DIR/handler.py
}

# ---------------------------------------------------------------------------- #
#                               Main Program                                   #
# ---------------------------------------------------------------------------- #

# start_nginx

echo "Pod Started"

setup_ssh
setup_sam3
setup_supersvg
setup_layeredsvg
setup_bezier
setup_longcat
setup_combined

case $MODE_TO_RUN in
    serverless)
        echo "Running in serverless mode"
        call_python_handler
        ;;
    pod)
        echo "Running in pod mode"
        start_sam3
        # start_supersvg
        # start_bezier
        start_combined
        ;;
    *)
        echo "Invalid MODE_TO_RUN value: $MODE_TO_RUN. Expected 'serverless', 'pod', or 'both'."
        exit 1
        ;;
esac

export_env_vars

echo "Start script(s) finished"

sleep infinity