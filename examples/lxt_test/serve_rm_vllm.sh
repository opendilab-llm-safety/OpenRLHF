#!/bin/bash
#SBATCH -p mllm-align
#SBATCH -N 1                       
#SBATCH --ntasks-per-node=1        
#SBATCH --gres=gpu:1               
#SBATCH -t 3-00:00:00             
#SBATCH --job-name=serve_rewardmodel
#SBATCH --quotatype=auto          
#SBATCH -o serve_rm_logs/output.%j.log    
#SBATCH -e serve_rm_logs/error.%j.log     

# 获取节点IP
NODE_IP=$(hostname -I | awk '{print $1}')
VLLM_PORT=20010
RM_SERVICE_PORT=20020
STATUS_FILE="rm_service_status.txt"
MODEL_PATH="/mnt/hwfile/llm-safety/models/huggingface/Qwen/Qwen2.5-72B-Instruct-AWQ"

# 激活conda环境
source /mnt/petrelfs/lixiangtian/miniconda3/etc/profile.d/conda.sh
conda activate rlhf

# 启动vLLM服务
echo "Starting vLLM service on ${NODE_IP}:${VLLM_PORT}..."
vllm serve \
    $MODEL_PATH \
    --port $VLLM_PORT \
    --host 0.0.0.0 \
    --served_model_name qwen &

# 等待vLLM服务启动
echo "Waiting for vLLM service to start..."
while ! curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null; do
    sleep 5
    echo "Still waiting for vLLM service..."
done
echo "vLLM service is ready!"

# 启动奖励模型服务
echo "Starting reward model service on ${NODE_IP}:${RM_SERVICE_PORT}..."
python -m openrlhf.cli.serve_rm_vllm \
    --model_host localhost \
    --model_port $VLLM_PORT \
    --model_name qwen \
    --host 0.0.0.0 \
    --port $RM_SERVICE_PORT \
    --batch_size 128 \
    --num_processes 128 \
    --max_len 2048 &


# 将服务信息写入状态文件
echo "RM_SERVICE_URL=http://${NODE_IP}:${RM_SERVICE_PORT}/get_reward" > $STATUS_FILE
echo "Service information saved to $STATUS_FILE"
echo "Reward Model Service URL: http://${NODE_IP}:${RM_SERVICE_PORT}/get_reward"

# 添加信号处理
cleanup() {
    echo "Cleaning up processes..."
    # 使用 pkill 发送 SIGTERM 信号，给进程一个优雅退出的机会
    pkill -TERM -f "vllm.entrypoints.openai.api_server"
    pkill -TERM -f "openrlhf.cli.serve_rm_vllm"
    
    # 等待几秒钟
    sleep 5
    
    # 如果进程还在运行，使用 SIGKILL 强制终止
    pkill -KILL -f "vllm.entrypoints.openai.api_server" 2>/dev/null
    pkill -KILL -f "openrlhf.cli.serve_rm_vllm" 2>/dev/null
    
    # 删除状态文件
    rm -f $STATUS_FILE
    echo "Cleanup completed"
    exit
}

# 注册清理函数
trap cleanup SIGTERM SIGINT

# 等待所有后台任务完成
wait