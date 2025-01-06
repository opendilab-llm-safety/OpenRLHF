#!/bin/bash
#SBATCH --partition=mllm-align
#SBATCH --job-name=vllm_deploy
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00
#SBATCH --output=vllm_deploy_%j.out
#SBATCH --error=vllm_deploy_%j.err

source /mnt/petrelfs/lixiangtian/miniconda3/bin/activate rlhf

MODEL_PATH="/mnt/hwfile/llm-safety/models/InternVL2_5-QwQ-38B-v5"
BASE_PORT=20005

# 在4个不同节点上分别启动服务
for i in {0..3}; do
    PORT=$((BASE_PORT + i * 10))
    
    # 为每个实例使用单独的输出文件
    srun --nodes=1 --ntasks=1 \
        --output=vllm_deploy_%j_node${i}.out \
        --error=vllm_deploy_%j_node${i}.err \
        bash -c "
        NODENAME=\$(hostname)
        IP=\$(hostname -i | awk '{print \$1}')
        
        echo \"Starting service on node: \$NODENAME\"
        echo \"Node ID: $i\"
        echo \"IP: \$IP\"
        echo \"Service URL: http://\${IP}:${PORT}/v1\"
        
        vllm serve \"$MODEL_PATH\" \
            --served-model-name intern_qwq_v5 \
            --tensor-parallel-size 4 \
            --port ${PORT} \
            --host 0.0.0.0
        " &
done

wait # 等待所有后台任务完成