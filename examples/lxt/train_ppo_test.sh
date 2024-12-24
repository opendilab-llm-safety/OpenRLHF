#!/bin/bash

#SBATCH -p mllm-align
#SBATCH -N 1                       # 使用1个节点
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:6               # 每个节点6张GPU
#SBATCH --mem=0                    # 全部内存
#SBATCH -t 3-00:00:00              # 运行3天
#SBATCH --job-name=ppo_test
#SBATCH --quotatype=auto           # auto模式
#SBATCH --mail-type=FAIL           # only send email on failure
#SBATCH -o train_ppo_logs/output.%j.log    
#SBATCH -e train_ppo_logs/error.%j.log   

# 项目设置
OPENRLHF_PATH='/mnt/petrelfs/lixiangtian/workspace/OpenRLHF'
RAY_VERSION=2.12.0

# Ray环境变量
export RAY_ADDRESS="http://127.0.0.1:20065"  # 与dashboard-port一致
export PYTHONPATH=$OPENRLHF_PATH:$PYTHONPATH

# 激活conda环境
source /mnt/petrelfs/lixiangtian/miniconda3/etc/profile.d/conda.sh
conda activate rlhf || {
    echo "Failed to activate conda environment 'rlhf'" >&2
    exit 1
}

JOBLOG="$(realpath .)/train_ppo_logs/train_ppo_test-$SLURM_JOB_ID.log"
echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..." &>> ${JOBLOG}

# 读取奖励模型服务信息
STATUS_FILE="rm_service_status.txt"
if [ ! -f "$STATUS_FILE" ]; then
    echo "Error: Reward model service information not found. Please start the reward model service first." &>> ${JOBLOG}
    exit 1
fi
source $STATUS_FILE

# Ray集群设置
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=( $nodes )
node_1=${nodes_array[0]}
ip=$node_1

# 清理可能存在的Ray进程
srun --nodes=1 --ntasks=1 -w "$node_1" bash -c "pkill -f ray" || true

port=20064
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"  &>> ${JOBLOG}

echo "STARTING HEAD at $node_1"  &>> ${JOBLOG}
echo "Starting Ray head node..." &>> ${JOBLOG}
ssh $node_1 "cd $OPENRLHF_PATH && \
    ray start --head \
    --node-ip-address=$ip \
    --port=$port \
    --dashboard-port=20065 \
    --dashboard-agent-grpc-port=20066 \
    --num-gpus=6" &>> ${JOBLOG}

# 等待Ray head node启动
MAX_RETRIES=30
RETRY_COUNT=0
while ! curl -s "http://localhost:20065/api/version" > /dev/null && [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    sleep 5
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Waiting for Ray head node... Attempt $RETRY_COUNT/$MAX_RETRIES" &>> ${JOBLOG}
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "Failed to start Ray head node" &>> ${JOBLOG}
    exit 1
fi

echo "Ray head node started successfully" &>> ${JOBLOG}

sleep 10s

worker_num=$((SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "STARTING WORKER $i at $node_i"  &>> ${JOBLOG}
    srun --nodes=1 --ntasks=1 -w "$node_i" bash -c \
        "cd $OPENRLHF_PATH \
        && ray start --address "$ip_head" --block" &>> ${JOBLOG} &
    sleep 5s
done

sleep 30s

# 检查是否有checkpoint
CHECKPOINT_DIR="/mnt/hwfile/llm-safety/checkpoints/Qwen2.5-0.5B-Instruct-PPO-MetaMathQA"
CHECKPOINT_FILE="pytorch_model.bin" # 指定需要加载的权重文件

echo "Using Reward Model Service: $RM_SERVICE_URL" &>> ${JOBLOG}
# PPO训练命令
ssh $node_1 \
    "cd $OPENRLHF_PATH && \
    ray job submit --address=http://localhost:20065 \
    --runtime-env-json='{
        "env_vars": {
            "WANDB_HTTP_PROXY": "http://lixiangtian:vJ7QF9Sx5qehJlqQga7XVTajpYbPKMG9WrhoURfD3Km8s7VacclbIqlCZnr1@10.1.20.50:23128",
            "WANDB_HTTPS_PROXY": "http://lixiangtian:vJ7QF9Sx5qehJlqQga7XVTajpYbPKMG9WrhoURfD3Km8s7VacclbIqlCZnr1@10.1.20.50:23128",
            "WANDB_API_KEY": "e8ab26345e839fdd0c5ca50a41be0c804bacd820",
            "WANDB_HTTP_TIMEOUT": "180"
        }
    }' \
    -- python -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 2 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 2 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 2 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules "q_proj" "k_proj" "v_proj" "o_proj" \
    --colocate_actor_ref \
    --pretrain /mnt/hwfile/llm-safety/models/huggingface/Qwen/Qwen2.5-0.5B-Instruct \
    --remote_rm_url $RM_SERVICE_URL \
    --save_path /mnt/hwfile/llm-safety/checkpoints/Qwen2.5-0.5B-Instruct-PPO-MetaMathQA \
    --input_key query \
    --reference_key response \
    --apply_chat_template \
    --micro_train_batch_size 4 \
    --train_batch_size 64 \
    --micro_rollout_batch_size 4 \
    --rollout_batch_size 128 \
    --max_epochs 1 \
    --prompt_max_len 10000 \
    --generate_max_len 20000 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data /mnt/hwfile/llm-safety/datasets/MetaMathQA/MetaMathQA-395K.json \
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
    --n_samples_per_prompt 1 \
    --advantage_estimator gae \
    --wandb_project ppo-training \
    --wandb_run_name Qwen2.5-0.5B-Instruct-PPO-MetaMathQA \
    --use_wandb ${WANDB_API_KEY} \
    ${CHECKPOINT_FILE:+--load_checkpoint "$CHECKPOINT_DIR/$CHECKPOINT_FILE"}" &>> ${JOBLOG} # 如果有checkpoint则加载

echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} finished ..." &>> ${JOBLOG}
