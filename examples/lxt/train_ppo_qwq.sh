#!/bin/bash

#SBATCH -p mllm-align
#SBATCH -N 2                       # 使用2个节点
#SBATCH --ntasks-per-node=1        
#SBATCH --gres=gpu:4               # 每个节点4张GPU
#SBATCH --mem=0                    # 全部内存
#SBATCH -t 3-00:00:00              # 运行3天
#SBATCH --job-name=ppo_38b
#SBATCH --quotatype=auto           # auto模式
#SBATCH -o output.%j.log            # 标准输出文件
#SBATCH -e error.%j.log             # 错误输出文件
#SBATCH --mail-type=FAIL           # only send email on failure
#SBATCH --overcommit               # needed for pytorch

# 项目设置
OPENRLHF_PATH=<OPENRLHF_ROOT_PATH>
MOUNT="$OPENRLHF_PATH:/openrlhf,$HOME/.cache:/root/.cache"
IMAGE_NAME="msr-aigc-sh/openrlhf_dev:5.0"
RAY_VERSION=2.9.3

JOBLOG="$(realpath .)/train_ppo_qwq-$SLURM_JOB_ID.log"
echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..." &>> ${JOBLOG}


# 读取奖励模型服务信息
STATUS_FILE="rm_service_status.txt"
if [ ! -f "$STATUS_FILE" ]; then
    echo "Error: Reward model service information not found. Please start the reward model service first." &>> ${JOBLOG}
    exit 1
fi
source $STATUS_FILE

# 环境设置
export MASTER_PORT=29610
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# Ray集群设置
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=( $nodes )
node_1=${nodes_array[0]}
ip=$node_1

port=20064
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"  &>> ${JOBLOG}

echo "STARTING HEAD at $node_1"  &>> ${JOBLOG}
srun --nodes=1 --ntasks=1 -w "$node_1" --container-image="$IMAGE_NAME" --container-mounts="$MOUNT" bash -c \
    "source /openrlhf/env.sh \
    && pip install ray[default]==$RAY_VERSION \
    && pip install openrlhf[vllm] \
    && /root/.local/bin/ray start --head --node-ip-address=$ip --port=$port --dashboard-port=20065 --dashboard-agent-grpc-port=20066 --block" &>> ${JOBLOG} &
sleep 30s

worker_num=$((SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "STARTING WORKER $i at $node_i"  &>> ${JOBLOG}
    srun --nodes=1 --ntasks=1 -w "$node_i" --container-image="$IMAGE_NAME" --container-mounts="$MOUNT" bash -c \
        "source /openrlhf/env.sh \
        && pip install ray[default]==$RAY_VERSION \
        && pip install openrlhf[vllm] \
        && /root/.local/bin/ray start --address "$ip_head" --block" &>> ${JOBLOG} &
    sleep 5s
done

sleep 30s

# 检查是否有checkpoint
CHECKPOINT_DIR="/mnt/hwfile/llm-safety/checkpoints/InternVL2_5-QwQ-38B-v5-PPO-MetaMathQA"
LATEST_CHECKPOINT=""
if [ -d "$CHECKPOINT_DIR" ]; then
    LATEST_CHECKPOINT=$(find "$CHECKPOINT_DIR" -name "checkpoint_*" -type d | sort -V | tail -n 1)
fi

echo "Using Reward Model Service: $RM_SERVICE_URL" &>> ${JOBLOG}
# PPO训练命令
srun --overlap --nodes=1 --ntasks=1 -w "$node_1" --container-image="$IMAGE_NAME" --container-mounts="$MOUNT" bash -c \
    "source /openrlhf/env.sh \
    && pip install ray[default]==$RAY_VERSION \
    && pip install openrlhf[vllm] \
    && /root/.local/bin/ray job submit --address=http://localhost:20065 \
        --runtime-env-json='{\"working_dir\": \"/openrlhf\", \"pip\": \"/openrlhf/requirements.txt\"}' \
        -- python -m openrlhf.cli.train_ppo_ray \
        --ref_num_nodes 1 \
        --ref_num_gpus_per_node 4 \
        --critic_num_nodes 1 \
        --critic_num_gpus_per_node 4 \
        --actor_num_nodes 1 \
        --actor_num_gpus_per_node 4 \
        --vllm_num_engines 1 \
        --vllm_tensor_parallel_size 4 \
        --lora_rank 8 \
        --lora_alpha 16 \
        --target_modules q_proj,k_proj,v_proj,o_proj \
        --colocate_actor_ref \
        --pretrain /mnt/hwfile/llm-safety/models/InternVL2_5-QwQ-38B-v5 \
        --remote_rm_url $RM_SERVICE_URL \
        --save_path /mnt/hwfile/llm-safety/checkpoints/InternVL2_5-QwQ-38B-v5-PPO-MetaMathQA \
        --input_key query \
        --reference_key response \
        --apply_chat_template \
        --micro_train_batch_size 2 \
        --train_batch_size 64 \
        --micro_rollout_batch_size 4 \
        --rollout_batch_size 256 \
        --max_epochs 1 \
        --prompt_max_len 32768 \
        --generate_max_len 8192 \
        --zero_stage 3 \
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
        --max_norm 1.0 \
        --eps_clip 0.2 \
        --value_clip 0.2 \
        --gamma 1.0 \
        --lambd 0.95 \
        --n_samples_per_prompt 1 \
        --advantage_estimator gae \
        --wandb_project ppo-training \
        --wandb_run_name InternVL2_5-QwQ-38B-v5-ppo \
        --use_wandb true \
        ${LATEST_CHECKPOINT:+--load_checkpoint "$LATEST_CHECKPOINT"}" &>> ${JOBLOG} # 如果有checkpoint则加载



echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} finished ..." &>> ${JOBLOG}
