#!/bin/bash

#SBATCH -p your-partition
#SBATCH -N 4                       # 使用4个节点
#SBATCH --ntasks-per-node=1        
#SBATCH --gpus-per-node=8         # 每个节点8张GPU
#SBATCH --mem=0                    
#SBATCH -t 7-00:00:00             # 运行7天
#SBATCH --job-name=ppo_38b

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
    --pretrain your-38b-model \
    --remote_rm_url http://localhost:5000/get_reward \  # 使用远程RM
    --save_path /path/to/save/checkpoint \
    --micro_train_batch_size 4 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 8 \
    --rollout_batch_size 512 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data your-prompt-dataset \
    --normalize_reward \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing \
    --packing_samples