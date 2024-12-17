#!/bin/bash

# 设置环境变量
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH=/mnt/petrelfs/lixiangtian/workspace/OpenRLHF:$PYTHONPATH
# 获取服务器IP地址
SERVER_IP=$(hostname -I | awk '{print $1}')
export RAY_ADDRESS=http://${SERVER_IP}:8265

# 创建日志目录
mkdir -p train_ppo_logs
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOGFILE="train_ppo_logs/train_ppo_qwq_${TIMESTAMP}.log"

# 日志函数
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1" | tee -a "${LOGFILE}"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" | tee -a "${LOGFILE}"
}

# 记录环境信息
log_info "=== Environment Information ==="
log_info "Hostname: $(hostname)"
log_info "Python version: $(python --version 2>&1)"
log_info "CUDA version: $(nvidia-smi | grep 'CUDA Version' | awk '{print $9}')"
log_info "GPU information:"
nvidia-smi >> "${LOGFILE}"

# 检查reward model服务
if [ ! -f "rm_service_status.txt" ]; then
    log_error "Reward model service status file not found!"
    exit 1
fi
source rm_service_status.txt
log_info "Using reward model service: ${RM_SERVICE_URL}"

# # 检查Ray集群状态
# log_info "Checking Ray cluster status..."
# # 使用curl检查Ray dashboard可用性
# if ! curl -s "$RAY_ADDRESS/api/version" > /dev/null; then
#     log_error "Cannot connect to Ray dashboard. Please ensure Ray cluster is running at $RAY_ADDRESS"
#     log_error "You can start it with: ray start --head --num-gpus=8 --dashboard-host=0.0.0.0"
#     exit 1
# fi
# log_info "Ray cluster is running and dashboard is accessible"

# 提交训练任务
log_info "Submitting training job..."
JOB_ID=$(ray job submit \
    --address="$RAY_ADDRESS" \
    --runtime-env-json='{"working_dir": "."}' \
    --no-wait \
    -- python -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 2 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 2 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 2 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 2 \
    --load_in_4bit \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules q_proj,k_proj,v_proj,o_proj \
    --colocate_actor_ref \
    --pretrain /mnt/hwfile/llm-safety/models/InternVL2_5-QwQ-38B-v5 \
    --remote_rm_url "${RM_SERVICE_URL}" \
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
    --vllm_sync_backend nccl \
    --max_norm 1.0 \
    --eps_clip 0.2 \
    --value_clip 0.2 \
    --gamma 1.0 \
    --lambd 0.95 \
    --n_samples_per_prompt 1 \
    --advantage_estimator gae \
    --wandb_project ppo-training \
    --wandb_run_name InternVL2_5-QwQ-38B-v5-ppo \
    --use_wandb "${WANDB_API_KEY}" 2>&1 | tee -a "${LOGFILE}" | grep -o 'JobSubmissionClient: Submitted Job.*' | cut -d' ' -f4)

if [ -n "$JOB_ID" ]; then
    log_info "Job submitted successfully with ID: $JOB_ID"
    
    # 监控任务状态
    while true; do
        STATUS=$(ray job status "$JOB_ID" 2>&1)
        log_info "Job status: $STATUS"
        
        if echo "$STATUS" | grep -q "SUCCEEDED"; then
            log_info "Job completed successfully"
            break
        elif echo "$STATUS" | grep -q "FAILED\|STOPPED\|PENDING"; then
            log_error "Job failed or stopped unexpectedly"
            ray job logs "$JOB_ID" >> "${LOGFILE}"
            exit 1
        fi
        
        sleep 30
    done
else
    log_error "Failed to submit job"
    exit 1
fi

log_info "Script completed"