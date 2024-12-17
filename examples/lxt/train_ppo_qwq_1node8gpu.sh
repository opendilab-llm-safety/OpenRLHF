#!/bin/bash

# 项目设置
OPENRLHF_PATH='/mnt/petrelfs/lixiangtian/workspace/OpenRLHF'
RAY_VERSION=2.12.0

# 创建日志目录
mkdir -p train_ppo_logs
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
JOBLOG="$(realpath .)/train_ppo_logs/train_ppo_qwq_${TIMESTAMP}.log"

# 日志函数
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1" | tee -a ${JOBLOG}
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" | tee -a ${JOBLOG}
}

log_debug() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [DEBUG] $1" | tee -a ${JOBLOG}
}

# 环境信息记录
log_info "=== Environment Information ==="
log_info "Hostname: $(hostname)"
log_info "Current directory: $(pwd)"
log_info "Python version: $(python --version 2>&1)"
log_info "Ray version: $(ray --version 2>&1)"
log_info "CUDA version: $(nvidia-smi | grep 'CUDA Version' | awk '{print $9}')"
log_info "GPU information:"
nvidia-smi &>> ${JOBLOG}
log_info "=========================="

# 清理已有Ray进程
log_info "Cleaning up existing Ray processes..."
ray stop &>> ${JOBLOG}
sleep 5
pkill -f ray &>> ${JOBLOG}
sleep 5

# 检查并清理Ray临时目录
RAY_TEMP_DIR="/tmp/ray_${TIMESTAMP}"
log_info "Cleaning up Ray temp directory: ${RAY_TEMP_DIR}"
if [ -d "${RAY_TEMP_DIR}" ]; then
    rm -rf "${RAY_TEMP_DIR}"
    log_debug "Removed existing Ray temp directory"
fi

# 读取奖励模型服务信息
STATUS_FILE="rm_service_status.txt"
if [ ! -f "$STATUS_FILE" ]; then
    log_error "Reward model service information not found. Please start the reward model service first."
    exit 1
fi
source $STATUS_FILE
log_info "Loaded reward model service configuration from ${STATUS_FILE}"
log_info "Reward Model URL: ${RM_SERVICE_URL}"

# Ray集群设置
ip=$(hostname -I | awk '{print $1}')
port=20064
ip_head=$ip:$port
export ip_head
log_info "Ray cluster head node: ${ip_head}"

# 添加项目路径到 PYTHONPATH
export PYTHONPATH=$OPENRLHF_PATH:$PYTHONPATH
log_info "PYTHONPATH: ${PYTHONPATH}"

# 启动Ray head节点
log_info "Starting Ray head node..."
ray start --head \
    --node-ip-address=$ip \
    --port=$port \
    --num-gpus=8 \
    --dashboard-port=20065 \
    --dashboard-agent-grpc-port=20066 \
    --temp-dir="${RAY_TEMP_DIR}" \
    &>> ${JOBLOG}

# 等待Ray服务就绪
log_info "Waiting for Ray runtime to be fully initialized..."
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if ray status &>/dev/null; then
        log_info "Ray cluster is ready!"
        break
    fi
    log_debug "Waiting for Ray cluster... Attempt $((RETRY_COUNT + 1))/$MAX_RETRIES"
    sleep 10
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    log_error "Ray cluster failed to initialize properly"
    ray stop &>> ${JOBLOG}
    exit 1
fi

# 检查Ray Dashboard可用性
log_info "Checking Ray Dashboard availability..."
DASHBOARD_URL="http://localhost:20065"
MAX_DASHBOARD_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_DASHBOARD_RETRIES ]; do
    if curl -s "${DASHBOARD_URL}/api/version" &>/dev/null; then
        log_info "Ray Dashboard is accessible at ${DASHBOARD_URL}"
        break
    fi
    log_debug "Waiting for Ray Dashboard... Attempt $((RETRY_COUNT + 1))/$MAX_DASHBOARD_RETRIES"
    sleep 10
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $MAX_DASHBOARD_RETRIES ]; then
    log_error "Ray Dashboard failed to become accessible"
    ray stop &>> ${JOBLOG}
    exit 1
fi

# 显示详细的Ray集群信息
log_info "=== Ray Cluster Status ==="
ray status &>> ${JOBLOG}
log_info "=========================="

# 检查GPU可见性
log_info "Checking GPU visibility for Ray..."
python -c "import ray; ray.init(); print(ray.get_gpu_ids())" &>> ${JOBLOG}

# 等待确保服务完全就绪
log_info "Waiting additional time for all services to stabilize..."
sleep 30

log_info "Preparing to submit training job..."
cd $OPENRLHF_PATH
log_info "Changed to directory: $(pwd)"

# PPO训练命令
log_info "Submitting training job..."
ray job submit --address="http://127.0.0.1:20065" \
    --runtime-env-json='{"working_dir": "."}' \
    -- python -u -m openrlhf.cli.train_ppo_ray \
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
    --use_wandb $WANDB_API_KEY \
    ${CHECKPOINT_FILE:+--load_checkpoint "$CHECKPOINT_DIR/$CHECKPOINT_FILE"} &>> ${JOBLOG}

JOB_STATUS=$?
if [ $JOB_STATUS -ne 0 ]; then
    log_error "Training job submission failed with status: ${JOB_STATUS}"
    # 收集额外的诊断信息
    log_debug "=== Ray Job Submission Diagnostics ==="
    log_debug "Ray Dashboard logs:"
    tail -n 50 "${RAY_TEMP_DIR}/logs/dashboard.log" &>> ${JOBLOG}
    log_debug "Ray GCS Server logs:"
    tail -n 50 "${RAY_TEMP_DIR}/logs/gcs_server.out" &>> ${JOBLOG}
    log_debug "=========================="
else
    log_info "Training job submitted successfully"
fi

log_info "Training process completed"
