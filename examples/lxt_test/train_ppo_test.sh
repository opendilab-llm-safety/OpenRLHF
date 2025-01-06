#!/bin/bash

#SBATCH -p mllm-align
#SBATCH -N 2                       # 使用2个节点
#SBATCH --ntasks-per-node=1         # 每个节点运行1个任务
#SBATCH --gres=gpu:2               # 每个节点2张GPU
#SBATCH --mem=0                    # 全部内存
#SBATCH -t 3-00:00:00              # 运行3天
#SBATCH --job-name=ppo_test
#SBATCH --quotatype=auto           # auto模式
#SBATCH --mail-type=FAIL           # only send email on failure
#SBATCH -o train_ppo_logs/output.%j.log    
#SBATCH -e train_ppo_logs/error.%j.log   
#SBATCH --exclude=SH-IDC1-10-140-1-68,SH-IDC1-10-140-0-134  # 排除指定节点

# 项目设置
OPENRLHF_PATH='/mnt/petrelfs/lixiangtian/workspace/OpenRLHF'
RAY_VERSION=2.12.0

# 动态生成端口号（从 30000 开始，避免与 Ray Worker 默认范围冲突）
export RAY_PORT=$((30000 + SLURM_JOB_ID % 10000))
export RAY_DASHBOARD_PORT=$((40000 + SLURM_JOB_ID % 10000))
export MIN_WORKER_PORT=$((50000 + SLURM_JOB_ID % 10000))
export MAX_WORKER_PORT=$((51000 + SLURM_JOB_ID % 10000))

# Ray 环境变量
export RAY_ADDRESS="http://127.0.0.1:$RAY_DASHBOARD_PORT"  # 与dashboard-port一致
export PYTHONPATH=$OPENRLHF_PATH:$PYTHONPATH

# 激活conda环境
source /mnt/petrelfs/lixiangtian/miniconda3/etc/profile.d/conda.sh
conda activate rlhf || {
    echo "Failed to activate conda environment 'rlhf'" >&2
    exit 1
}

# 日志文件
JOBLOG="$(realpath .)/train_ppo_logs/train_ppo_test-$SLURM_JOB_ID.log"
echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..." &>> ${JOBLOG}

# 读取奖励模型服务信息
STATUS_FILE="rm_service_status.txt"
if [ ! -f "$STATUS_FILE" ]; then
    echo "Error: Reward model service information not found. Please start the reward model service first." &>> ${JOBLOG}
    exit 1
fi
source $STATUS_FILE

# 获取节点IP地址列表
get_node_ip() {
    local node=$1
    srun --nodes=1 --ntasks=1 -w "$node" hostname -i | awk '{print $1}'
}

# 获取节点列表
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
node_1_hostname=${nodes_array[0]}
node_2_hostname=${nodes_array[1]}

# 获取节点IP地址
node_1_ip=$(get_node_ip "$node_1_hostname")
node_2_ip=$(get_node_ip "$node_2_hostname")

# 停止可能存在的 Ray 进程
echo "Stopping existing Ray processes..." &>> ${JOBLOG}
srun --nodes=2 --ntasks=2 bash -c "ray stop --force" || true

# 修改环境变量的设置方式，确保在所有节点上可用
export_vars="export RAY_PORT=$RAY_PORT; \
             export RAY_DASHBOARD_PORT=$RAY_DASHBOARD_PORT; \
             export MIN_WORKER_PORT=$MIN_WORKER_PORT; \
             export MAX_WORKER_PORT=$MAX_WORKER_PORT; \
             export RAY_ADDRESS=$RAY_ADDRESS; \
             export PYTHONPATH=$OPENRLHF_PATH:\$PYTHONPATH"

# 启动 Ray 头节点时先设置环境变量
echo "Starting Ray head node on $node_1_ip" &>> ${JOBLOG}
srun --nodes=1 --ntasks=1 -w "$node_1_hostname" bash -c \
    "$export_vars && \
    cd $OPENRLHF_PATH && \
    ray stop --force && \
    ray start --head \
    --node-ip-address=$node_1_ip \
    --port=$RAY_PORT \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=$RAY_DASHBOARD_PORT \
    --min-worker-port=$MIN_WORKER_PORT \
    --max-worker-port=$MAX_WORKER_PORT \
    --num-gpus=2" &>> ${JOBLOG}

# 等待 Ray 头节点完全启动
sleep 20

# 启动 Ray 工作节点时也设置环境变量，并增加重试机制
ip_head="$node_1_ip:$RAY_PORT"
echo "Starting Ray worker node on $node_2_ip" &>> ${JOBLOG}

MAX_WORKER_RETRIES=3
for attempt in $(seq 1 $MAX_WORKER_RETRIES); do
    echo "Worker node connection attempt $attempt/$MAX_WORKER_RETRIES" &>> ${JOBLOG}
    
    srun --nodes=1 --ntasks=1 -w "$node_2_hostname" bash -c \
        "$export_vars && \
        cd $OPENRLHF_PATH && \
        ray stop --force && \
        ray start --address=$ip_head \
        --min-worker-port=$MIN_WORKER_PORT \
        --max-worker-port=$MAX_WORKER_PORT \
        --num-gpus=2" &>> ${JOBLOG}
    
    # 检查worker是否成功连接
    sleep 10
    if srun --nodes=1 --ntasks=1 -w "$node_2_hostname" ray status --address=$ip_head &>> ${JOBLOG}; then
        echo "Worker node successfully connected" &>> ${JOBLOG}
        break
    else
        echo "Worker node connection failed, retrying..." &>> ${JOBLOG}
        if [ $attempt -eq $MAX_WORKER_RETRIES ]; then
            echo "Failed to connect worker node after $MAX_WORKER_RETRIES attempts" &>> ${JOBLOG}
            exit 1
        fi
        sleep 5
    fi
done

# 等待 Ray 头节点启动
MAX_RETRIES=30
RETRY_COUNT=0
while ! curl -s "http://$node_1_ip:$RAY_DASHBOARD_PORT/api/version" > /dev/null && [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    sleep 5
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Waiting for Ray head node... Attempt $RETRY_COUNT/$MAX_RETRIES" &>> ${JOBLOG}
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "Failed to start Ray head node" &>> ${JOBLOG}
    exit 1
fi

echo "Ray head node started successfully" &>> ${JOBLOG}

# 检查checkpoint目录
CHECKPOINT_DIR="/mnt/hwfile/llm-safety/checkpoints/Qwen2.5-0.5B-Instruct-PPO-MetaMathQA"
CHECKPOINT_FILE="model.safetensors"
CHECKPOINT_PATH="$CHECKPOINT_DIR/$CHECKPOINT_FILE"

# 准备load_checkpoint参数
LOAD_CHECKPOINT_ARG=""
if [ -f "$CHECKPOINT_PATH" ]; then
    LOAD_CHECKPOINT_ARG="--load_checkpoint $CHECKPOINT_PATH"
fi

echo "Using Reward Model Service: $RM_SERVICE_URL" &>> ${JOBLOG}

# 提交任务
echo "Submitting PPO training job to Ray cluster" &>> ${JOBLOG}
srun --nodes=1 --ntasks=1 -w "$node_1_hostname" bash -c \
    "cd $OPENRLHF_PATH && \
    ray job submit --address=http://$node_1_ip:$RAY_DASHBOARD_PORT \
    --runtime-env-json="{
        \"env_vars\": {
            \"WANDB_HTTP_PROXY\": \"http://lixiangtian:vJ7QF9Sx5qehJlqQga7XVTajpYbPKMG9WrhoURfD3Km8s7VacclbIqlCZnr1@10.1.20.50:23128\",
            \"WANDB_HTTPS_PROXY\": \"http://lixiangtian:vJ7QF9Sx5qehJlqQga7XVTajpYbPKMG9WrhoURfD3Km8s7VacclbIqlCZnr1@10.1.20.50:23128\",
            \"WANDB_API_KEY\": \"e8ab26345e839fdd0c5ca50a41be0c804bacd820\",
            \"WANDB_HTTP_TIMEOUT\": \"180\"
        }
    }" \
    -- python -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 1 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 2 \
    --colocate_actor_ref \
    --pretrain /mnt/hwfile/llm-safety/models/huggingface/Qwen/Qwen2-VL-7B-Instruct \
    --remote_rm_url $RM_SERVICE_URL \
    --save_path /mnt/hwfile/llm-safety/checkpoints/Qwen2-VL-7B-Instruct-PPO-MetaMathQA \
    --input_key query \
    --reference_key response \
    --apply_chat_template \
    --micro_train_batch_size 32 \
    --train_batch_size 32 \
    --micro_rollout_batch_size 32 \
    --rollout_batch_size 32 \
    --max_epochs 1 \
    --prompt_max_len 10000 \
    --generate_max_len 20000 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data /mnt/hwfile/llm-safety/datasets/MetaMathQA/MetaMathQA-395K-1000.json \
    --normalize_reward \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing \
    --packing_samples \
    --vllm_sync_backend nccl \
    --max_norm 1.0 \
    --eps_clip 0.2 \
    --value_clip 0.2 \
    --gamma 1.0 \
    --lambd 0.95 \
    --n_samples_per_prompt 3 \
    --advantage_estimator gae \
    --wandb_project ppo-training \
    --wandb_run_name Qwen2-VL-7B-Instruct-PPO-MetaMathQA \
    --use_wandb ${WANDB_API_KEY} \
    $LOAD_CHECKPOINT_ARG &>> ${JOBLOG}

# 停止 Ray 进程
echo "Stopping Ray processes..." &>> ${JOBLOG}
srun --nodes=2 --ntasks=2 bash -c "ray stop --force" || true

echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} finished ..." &>> ${JOBLOG}
