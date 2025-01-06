# PPO训练手动执行步骤指南

# 1. 环境变量设置（在两个节点上执行）
export OPENRLHF_PATH='/mnt/petrelfs/lixiangtian/workspace/OpenRLHF'
export PYTHONPATH=$OPENRLHF_PATH:$PYTHONPATH
export RAY_VERSION=2.12.0
export RAY_PORT=30001  # 示例端口号，可根据实际情况调整
export RAY_DASHBOARD_PORT=40001
export MIN_WORKER_PORT=51001
export MAX_WORKER_PORT=52001
# 设置主节点IP,在实际执行时需要替换为实际的头节点IP
export MASTER_NODE_IP=$(hostname -i)  # 在头节点上执行
echo "MASTER_NODE_IP: $MASTER_NODE_IP"
# export MASTER_NODE_IP="10.140.0.134"  # 在工作节点上执行时取消注释并填入头节点IP

export RAY_ADDRESS="http://${MASTER_NODE_IP}:$RAY_DASHBOARD_PORT"
echo "RAY_ADDRESS: $RAY_ADDRESS"

# 读取远程奖励模型url
# RM_SERVICE_URL=http://10.140.1.175:20020/get_reward
source rm_service_status.txt
echo $RM_SERVICE_URL

# 2. 激活conda环境（在两个节点上执行）
source /mnt/petrelfs/lixiangtian/miniconda3/etc/profile.d/conda.sh
conda activate rlhf

# 3. 在头节点（SH-IDC1-10-140-0-134）启动Ray服务
# 在头节点终端执行以下命令：
ray stop --force
ray start --head \
    --node-ip-address=$MASTER_NODE_IP \
    --port=$RAY_PORT \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=$RAY_DASHBOARD_PORT \
    --min-worker-port=$MIN_WORKER_PORT \
    --max-worker-port=$MAX_WORKER_PORT \
    --num-gpus=2

# 4. 在工作节点（SH-IDC1-10-140-1-98）加入Ray集群
# 在工作节点终端执行以下命令：
ray stop --force
ray start --address=${MASTER_NODE_IP}:$RAY_PORT \
    --min-worker-port=$MIN_WORKER_PORT \
    --max-worker-port=$MAX_WORKER_PORT \
    --num-gpus=2

# 5. 在头节点提交训练任务
# 在头节点终端执行以下命令：
cd $OPENRLHF_PATH
ray job submit --address=http://${MASTER_NODE_IP}:$RAY_DASHBOARD_PORT \
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
    --vllm_tensor_parallel_size 1 \
    --colocate_actor_ref \
    --pretrain /mnt/hwfile/llm-safety/models/huggingface/Qwen/Qwen2.5-0.5B-Instruct \
    --remote_rm_url $RM_SERVICE_URL \
    --save_path /mnt/hwfile/llm-safety/checkpoints/Qwen2.5-0.5B-Instruct-PPO-MetaMathQA \
    --input_key query \
    --reference_key response \
    --apply_chat_template \
    --micro_train_batch_size 64 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 64 \
    --rollout_batch_size 128 \
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
    --n_samples_per_prompt 1 \
    --advantage_estimator gae \
    --wandb_project ppo-training \
    --wandb_run_name Qwen2.5-0.5B-Instruct-PPO-MetaMathQA \
    --use_wandb ${WANDB_API_KEY}

# 6. 训练完成后停止Ray服务（在两个节点上执行）
ray stop --force

# 7. 检查训练日志
# 日志文件位于：train_ppo_logs/output.*.log
