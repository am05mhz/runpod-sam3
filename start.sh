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
    if [[ $PUBLIC_KEY ]]; then
        echo "Setting up SSH..."
        mkdir -p ~/.ssh
        echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
        chmod 700 -R ~/.ssh
        # Generate SSH host keys if not present
        generate_ssh_keys
        service ssh start
        echo "SSH host keys:"
        cat /etc/ssh/*.pub
    fi
}

setup_sam3() {
    echo "installing sam3..."
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
    echo "installing supersvg..."
    source /app/venv/apps/bin/activate
    if [ ! -d "/workspace/apps" ]; then
        mkdir -p /workspace/apps
    fi
    cd /workspace/apps
    if [ ! -d "/workspace/apps/supersvg" ]; then
        git clone https://github.com/sjtuplayer/SuperSVG.git supersvg
    fi
    cp -r /app/supersvg /workspace/apps/
    cd supersvg
    pip install --upgrade --no-cache-dir -r requirements.txt
    cd DiffVG
    git submodule update --init --recursive
    python setup.py install
}

setup_bezier() {
    echo "installing bezier splatting..."
    source /app/venv/bezier/bin/activate
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
    if [ ! -d "/workspace/apps/bezier/gsplat" ]; then
        git clone https://github.com/XingtongGe/gsplat.git
    fi
    cd /workspace/apps/bezier/gsplat
    ls -la
    pip install -e .
}

start_sam3() {
    echo "starting sam3..."
    source /app/venv/apps/bin/activate
    cd /workspace/apps/sam3
    nohup python server.py --port=$SAM3_PORT > /proc/self/fd/1 2>&1 &
}

start_supersvg() {
    if [[ $START_SUPERSVG ]]; then
        echo "starting supersvg..."
        source /app/venv/apps/bin/activate
        cd /workspace/apps/supersvg
        nohup python server.py --port=$SUPERSVG_PORT > /proc/self/fd/1 2>&1 &
    fi
}

start_bezier() {
    if [[ $START_BEZIER ]]; then
        echo "starting bezier splatting..."
        source /app/venv/bezier/bin/activate
        cd /workspace/apps/bezier
        nohup python server.py --port=$BEZIER_PORT > /proc/self/fd/1 2>&1 &
    fi
}

setup_vllm() {
    echo "setup vllm..."
    if [[ $USE_VLLM ]]; then
        pip install vllm --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128
        echo "starting vllm..."
        vllm serve Qwen/Qwen3-VL-8B-Thinking --max-num-seqs 2 --tensor-parallel-size 1 --gpu-memory-utilization 0.55 --allowed-local-media-path / --enforce-eager --port $VLLM_PORT
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

start_nginx

echo "Pod Started"

setup_ssh
setup_sam3
setup_supersvg
setup_bezier

case $MODE_TO_RUN in
    serverless)
        echo "Running in serverless mode"
        call_python_handler
        ;;
    pod)
        echo "Running in pod mode"
        start_sam3
        start_supersvg
        start_bezier
        ;;
    *)
        echo "Invalid MODE_TO_RUN value: $MODE_TO_RUN. Expected 'serverless', 'pod', or 'both'."
        exit 1
        ;;
esac

export_env_vars

echo "Start script(s) finished"

sleep infinity