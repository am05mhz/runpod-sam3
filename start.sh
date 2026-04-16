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
    if [ ! -d "$RP_WORKSPACE/apps" ]; then
        mkdir -p "$RP_WORKSPACE/apps"
    fi
    cp -r /app/sam3 $RP_WORKSPACE/apps/
}

setup_supersvg() {
    echo ""
    echo "========================================================"
    echo "installing supersvg..."
    echo "========================================================"
    if [ ! -d "$RP_WORKSPACE/apps" ]; then
        mkdir -p $RP_WORKSPACE/apps
    fi
    cp -r /app/supersvg $RP_WORKSPACE/apps/
}

setup_combined() {
    echo ""
    echo "========================================================"
    echo "installing combined api..."
    echo "========================================================"
    cp /app/combined_api.py $RP_WORKSPACE/apps/
    cp /app/common_utils.py $RP_WORKSPACE/apps/
    cp -r /app/templates $RP_WORKSPACE/apps/
}

setup_layeredsvg() {
    echo ""
    echo "========================================================"
    echo "installing layeredsvg..."
    echo "========================================================"
    if [ ! -d "$RP_WORKSPACE/apps" ]; then
        mkdir -p $RP_WORKSPACE/apps
    fi
    cp -r /app/layeredsvg $RP_WORKSPACE/apps/
}

setup_longcat() {
    echo ""
    echo "========================================================"
    echo "installing longcat..."
    echo "========================================================"
    source /app/venv/apps/bin/activate
    if [ ! -d "$RP_WORKSPACE/apps" ]; then
        mkdir -p $RP_WORKSPACE/apps
    fi
    cp -r /app/longcat $RP_WORKSPACE/apps/
}

setup_bezier() {
    echo ""
    echo "========================================================"
    echo "installing bezier splatting..."
    echo "========================================================"
    source /app/venv/apps/bin/activate
    if [ ! -d "$RP_WORKSPACE/apps" ]; then
        mkdir -p $RP_WORKSPACE/apps
    fi
    cp -r /app/bezier $RP_WORKSPACE/apps/
}

start_sam3() {
    echo ""
    echo "========================================================"
    echo "starting sam3..."
    echo "========================================================"
    source /app/venv/apps/bin/activate
    cd $RP_WORKSPACE/apps/sam3
    nohup python server.py --port=$SAM3_PORT > /proc/self/fd/1 2>&1 &
}

start_supersvg() {
    if [[ $START_SUPERSVG ]]; then
        echo ""
        echo "========================================================"
        echo "starting supersvg..."
        echo "========================================================"
        source /app/venv/apps/bin/activate
        cd $RP_WORKSPACE/apps/supersvg
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
        cd $RP_WORKSPACE/apps
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
        cd $RP_WORKSPACE/apps/bezier
        nohup python server.py --port=$BEZIER_PORT > /proc/self/fd/1 2>&1 &
    else
        echo "not starting bezier splat"
    fi
}

# setup_vllm() {
#     echo ""
#     echo "========================================================"
#     echo "setup vllm..."
#     if [[ $USE_VLLM ]]; then
#         pip install vllm --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128
#         echo "starting vllm..."
#         vllm serve Qwen/Qwen3-VL-8B-Thinking --max-num-seqs 2 --tensor-parallel-size 1 --gpu-memory-utilization 0.55 --allowed-local-media-path / --enforce-eager --port $VLLM_PORT
#     else
#         echo "not setup vllm"
#     fi
# }

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
    python $WORKSPACE_DIR/handler_runpod.py
}

copy_setup() {
    echo "copying apps..."
    if [ ! -d "$RP_WORKSPACE/apps" ]; then
        mkdir -p $RP_WORKSPACE/apps
    fi
    cp -r $WORKSPACE_DIR/* $RP_WORKSPACE/apps/
}

# ---------------------------------------------------------------------------- #
#                               Main Program                                   #
# ---------------------------------------------------------------------------- #

# start_nginx

echo "Pod Started"

setup_ssh
copy_setup
# setup_sam3
# setup_supersvg
# setup_layeredsvg
# setup_bezier
# setup_longcat
# setup_combined

case $MODE_TO_RUN in
    serverless)
        echo "Running in serverless mode"
        call_python_handler
        ;;
    pod)
        echo "Running in pod mode"
        # start_sam3
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