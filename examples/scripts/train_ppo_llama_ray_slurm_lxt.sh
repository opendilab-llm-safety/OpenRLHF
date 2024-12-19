#!/bin/bash

#SBATCH -p mllm-align
#SBATCH -J ppo_ray_slurm
#SBATCH -N 1                       # 64x8x4
#SBATCH -t 3-00:00:00              # wall time
#SBATCH --ntasks-per-node=1        # tasks per node
#SBATCH --exclusive                # exclusive node access
#SBATCH --mem=0                    # all mem avail
#SBATCH --mail-type=FAIL           # only send email on failure
#SBATCH --overcommit               # needed for pytorch

# project settings
OPENRLHF_PATH=/mnt/petrelfs/lixiangtian/workspace/OpenRLHF
MOUNT="$OPENRLHF_PATH:/openrlhf,$HOME/.cache:/root/.cache"
IMAGE_NAME="nvcr.io/nvidia/pytorch:24.07-py3"
RAY_VERSION=2.12.0

JOBLOG="$(realpath .)/train_ppo_llama_ray-$SLURM_JOB_ID.log"
echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..." &>> ${JOBLOG}

# launch ray daemon
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST") # Getting the node names
nodes_array=( $nodes )
node_1=${nodes_array[0]}
ip=$node_1

port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"  &>> ${JOBLOG}

echo "STARTING HEAD at $node_1"  &>> ${JOBLOG}
# Pull image and start head node
srun --nodes=1 --ntasks=1 -w "$node_1" bash -c "apptainer pull pytorch.sif docker://$IMAGE_NAME && apptainer exec --nv --bind \"$OPENRLHF_PATH:/openrlhf,$HOME/.cache:/root/.cache\" pytorch.sif bash -c 'pip install ray[default]==$RAY_VERSION && /root/.local/bin/ray start --head --node-ip-address=$ip --port=$port --block'" &>> ${JOBLOG} &
sleep 10s

worker_num=$((SLURM_JOB_NUM_NODES)) #number of nodes other than the head node
for ((i = 1; i < worker_num; i++)); do
node_i=${nodes_array[$i]}
echo "STARTING WORKER $i at $node_i"  &>> ${JOBLOG}
# Pull image and start worker node
srun --nodes=1 --ntasks=1 -w "$node_i" bash -c "apptainer pull pytorch.sif docker://$IMAGE_NAME && apptainer exec --nv --bind \"$OPENRLHF_PATH:/openrlhf,$HOME/.cache:/root/.cache\" pytorch.sif bash -c 'pip install ray[default]==$RAY_VERSION && /root/.local/bin/ray start --address \"$ip_head\" --block'" &>> ${JOBLOG} &
sleep 1s;
done

sleep 30s

# ===== submit ray job =====
# Job start
srun --overlap --nodes=1 --ntasks=1 -w "$node_1" bash -c "apptainer exec --nv --bind \"$OPENRLHF_PATH:/openrlhf,$HOME/.cache:/root/.cache\" pytorch.sif bash -c 'pip install ray[default]==$RAY_VERSION && /root/.local/bin/ray job submit --address=http://localhost:8265 --runtime-env-json=\"{\\\"working_dir\\\": \\\"/openrlhf\\\", \\\"pip\\\": \\\"/openrlhf/requirements.txt\\\"}\" -- python3 -m openrlhf.cli.train_ppo_ray --ref_num_nodes 1 --ref_num_gpus_per_node 2 --reward_num_nodes 1 --reward_num_gpus_per_node 2 --critic_num_nodes 1 --critic_num_gpus_per_node 2 --actor_num_nodes 1 --actor_num_gpus_per_node 2 --vllm_num_engines 1 --vllm_tensor_parallel_size 4 --colocate_critic_reward --colocate_actor_ref --pretrain OpenRLHF/Llama-3-8b-sft-mixture --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture --save_path /openrlhf/examples/checkpoint/llama3-8b-rlhf --micro_train_batch_size 8 --train_batch_size 128 --micro_rollout_batch_size 16 --rollout_batch_size 1024 --max_samples 100000 --max_epochs 1 --prompt_max_len 1024 --generate_max_len 1024 --zero_stage 3 --bf16 --actor_learning_rate 5e-7 --critic_learning_rate 9e-6 --init_kl_coef 0.01 --prompt_data OpenRLHF/prompt-collection-v0.1 --input_key context_messages --apply_chat_template --normalize_reward --adam_offload --flash_attn --packing_samples --vllm_sync_backend nccl --gradient_checkpointing --use_wandb {wandb_token}'" &>> ${JOBLOG}

echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..." &>> ${JOBLOG}
