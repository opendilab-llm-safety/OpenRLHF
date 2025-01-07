#!/bin/bash

#SBATCH -p mllm-align
#SBATCH -N 1                       
#SBATCH --ntasks-per-node=1        
#SBATCH --gres=gpu:1               
#SBATCH -t 3-00:00:00             
#SBATCH --job-name=serve_rewardmodel
#SBATCH --quotatype=auto          
#SBATCH -o serve_rm_logs/output.%j.log    

# 获取节点IP
NODE_IP=$(hostname -I | awk '{print $1}')
RM_MODEL_PORT=10095
RM_SERVICE_PORT=10100
STATUS_FILE="rm_service_status.txt"

# 定义检查服务是否就绪的函数
check_service() {
    local port=$1
    local max_attempts=$2
    local attempt=0
    
    echo "Checking service on port $port..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://127.0.0.1:${port}/get_model_info" > /dev/null; then
            echo "Service on port $port is ready!"
            return 0
        fi
        attempt=$((attempt + 1))
        echo "Waiting for service to be ready... Attempt $attempt/$max_attempts"
        sleep 10
    done
    
    echo "Service on port $port failed to start after $max_attempts attempts"
    return 1
}

# 激活conda环境
source /mnt/petrelfs/lixiangtian/miniconda3/etc/profile.d/conda.sh
conda activate sglang || {
    echo "Failed to activate conda environment 'rlhf'" >&2
    exit 1
}

# 启动基础模型服务
echo "Starting base model service on ${NODE_IP}:${RM_MODEL_PORT}..."
python -m sglang.launch_server \
    --model-path /mnt/hwfile/llm-safety/models/huggingface/Qwen/Qwen2.5-14B-Instruct-AWQ \
    --host 0.0.0.0 \
    --port $RM_MODEL_PORT 2>&1 &

sleep 100
# 等待基础模型服务完全启动（最多等待30次，每次10秒）
if ! check_service $RM_MODEL_PORT 30; then
    echo "Failed to start base model service"
    exit 1
fi

# 启动奖励模型服务
echo "Starting reward model service on ${NODE_IP}:${RM_SERVICE_PORT}..."
python -m openrlhf.cli.serve_rm_lxt \
    --host 0.0.0.0 \
    --port $RM_SERVICE_PORT \
    --model_endpoint http://127.0.0.1:$RM_MODEL_PORT \
    --batch_size 16 \
    --max_len 32768 2>&1 &

sleep 5

# 将服务信息写入状态文件
echo "RM_SERVICE_URL=http://${NODE_IP}:${RM_SERVICE_PORT}/get_reward" > $STATUS_FILE
echo "Service information saved to $STATUS_FILE"
echo "Reward Model Service URL: http://${NODE_IP}:${RM_SERVICE_PORT}/get_reward"

# 添加信号处理
cleanup() {
    echo "Cleaning up processes..."
    pkill -f "sglang.launch_server"
    pkill -f "openrlhf.cli.serve_rm_lxt"
    rm -f $STATUS_FILE
    exit
}

# 注册清理函数
trap cleanup SIGTERM SIGINT

# 等待所有后台任务完成
wait
