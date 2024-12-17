#!/bin/bash

# 设置环境变量 
export PYTHONPATH=/mnt/petrelfs/lixiangtian/workspace/OpenRLHF:$PYTHONPATH

# 创建日志目录
mkdir -p train_ppo_logs
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOGFILE="train_ppo_logs/train_ppo_qwq_normal_${TIMESTAMP}.log"

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

# 启动训练
log_info "Starting PPO training..."

deepspeed --module openrlhf.cli.train_ppo \
    --pretrain /mnt/hwfile/llm-safety/models/InternVL2_5-QwQ-38B-v5 \
    --remote_rm_url "${RM_SERVICE_URL}" \
    --save_path /mnt/hwfile/llm-safety/checkpoints/InternVL2_5-QwQ-38B-v5-PPO-MetaMathQA \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
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
    --input_key query \
    --reference_key response \
    --apply_chat_template \
    --normalize_reward \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules "language_model.model.layers.*.self_attn.q_proj,language_model.model.layers.*.self_attn.k_proj,language_model.model.layers.*.self_attn.v_proj,language_model.model.layers.*.self_attn.o_proj" \
    --max_norm 1.0 \
    --eps_clip 0.2 \
    --value_clip 0.2 \
    --gamma 1.0 \
    --lambd 0.95 \
    --advantage_estimator gae \
    --wandb_project ppo-training \
    --wandb_run_name InternVL2_5-QwQ-38B-v5-ppo \
    --use_wandb "${WANDB_API_KEY}" 2>&1 | tee -a "${LOGFILE}"

training_status=$?

if [ $training_status -eq 0 ]; then
    log_info "Training completed successfully"
else
    log_error "Training failed with status $training_status"
    exit 1
fi

log_info "Script completed"