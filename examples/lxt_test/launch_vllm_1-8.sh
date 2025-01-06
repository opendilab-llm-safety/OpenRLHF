#!/bin/bash
#SBATCH --partition=mllm-align
#SBATCH --job-name=vllm_deploy
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=3-00:00:00
#SBATCH --output=vllm_deploy_%j.out
#SBATCH --error=vllm_deploy_%j.err

# 初始化conda
source /mnt/petrelfs/lixiangtian/miniconda3/bin/activate rlhf

# 模型路径
MODEL_PATH="/mnt/hwfile/llm-safety/models/huggingface/Qwen/Qwen2.5-14B-Instruct-AWQ"
BASE_PORT=10005

# 启动8个独立实例
for i in {0..7}; do
    PORT=$((BASE_PORT + i))
    
    # 为每个实例使用单独的输出文件
    nohup bash -c "
    NODENAME=\$(hostname)
    IP=\$(hostname -i | awk '{print \$1}')
    
    echo \"Starting service on GPU: $i\"
    echo \"Node: \$NODENAME\"
    echo \"IP: \$IP\"
    echo \"Service URL: http://\${IP}:${PORT}/v1\"
    
    CUDA_VISIBLE_DEVICES=$i vllm serve \"$MODEL_PATH\" \
        --served-model-name qwen \
        --tensor-parallel-size 1 \
        --port ${PORT} \
        --host 0.0.0.0 \
        --gpu-memory-utilization 0.9
    " > vllm_deploy_${SLURM_JOB_ID}_gpu${i}.out 2>&1 &
done

wait # 等待所有后台任务完成

# 使用方法：
# 1. 提交作业：
#    ```bash
#    sbatch examples/lxt_test/launch_vllm_1-8.sh
#    ```

# 2. 查看作业状态：
#    ```bash
#    squeue -u $USER
#    ```

# 3. 查看日志：
#    ```bash
#    tail -f vllm_deploy_*_gpu*.out
#    ```
