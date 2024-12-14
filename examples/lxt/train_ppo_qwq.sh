#!/bin/bash

#SBATCH -p mllm-align
#SBATCH -N 4                       # 使用4个节点
#SBATCH --ntasks-per-node=1        
#SBATCH --gres=gpu:8         # 每个节点8张GP            
#SBATCH -t 7-00:00:00             # 运行7天
#SBATCH --job-name=ppo_38b
#SBATCH --comment="auto"          # auto模式
#SBATCH -o output.%j.log    # 标准输出文件
#SBATCH -e error.%j.log     # 错误输出文件

# 环境设置
export MASTER_PORT=29500
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# Ray集群设置
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=( $nodes )
node_1=${nodes_array[0]}
ip_head=$node_1:6379

# 启动Ray head节点
srun --nodes=1 --ntasks=1 -w "$node_1" \
    ray start --head --node-ip-address=$node_1 --port=6379 --block &
sleep 10

# 启动Ray worker节点
for ((i = 1; i < SLURM_JOB_NUM_NODES; i++)); do
    node_i=${nodes_array[$i]}
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" --block &
    sleep 5
done

# 检查是否有checkpoint
CHECKPOINT_DIR="/mnt/hwfile/llm-safety/checkpoints/InternVL2_5-QwQ-38B-v5-PPO-MetaMathQA"
LATEST_CHECKPOINT=""
if [ -d "$CHECKPOINT_DIR" ]; then
    LATEST_CHECKPOINT=$(find "$CHECKPOINT_DIR" -name "checkpoint_*" -type d | sort -V | tail -n 1)
fi

# PPO训练命令
srun --overlap --nodes=1 --ntasks=1 -w "$node_1" \
python -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 8 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 8 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 8 \
    --vllm_num_engines 8 \
    --vllm_tensor_parallel_size 4 \
    --colocate_actor_ref \
    --pretrain /mnt/hwfile/llm-safety/models/InternVL2_5-QwQ-38B-v5 \
    --remote_rm_url http://10.1.96.87:30000/get_reward \
    --save_path /mnt/hwfile/llm-safety/checkpoints/InternVL2_5-QwQ-38B-v5-PPO-MetaMathQA \
    --input_key "query" \
    --reference_key "response" \
    --apply_chat_template \
    --micro_train_batch_size 4 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 8 \
    --rollout_batch_size 512 \
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
    --reward_clip_range 5 \
    --wandb_project "ppo-training" \
    --wandb_run_name "InternVL2_5-QwQ-38B-v5-ppo" \
    --use_wandb true \
    ${LATEST_CHECKPOINT:+--load_checkpoint "$LATEST_CHECKPOINT"} # 如果有checkpoint则加载