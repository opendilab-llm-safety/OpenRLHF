#!/bin/bash

#SBATCH -p mllm-align
#SBATCH -N 1                       # 使用1个节点
#SBATCH --ntasks-per-node=1        
#SBATCH --gres=gpu:1         # 奖励模型只需要1个GPU                
#SBATCH -t 7-00:00:00             # 运行7天
#SBATCH --job-name=serve_rm
#SBATCH --comment="auto"          # auto模式
#SBATCH -o output.%j.log    # 标准输出文件
#SBATCH -e error.%j.log     # 错误输出文件

# 环境设置
# export CUDA_VISIBLE_DEVICES=0

# 激活 sglang 环境
conda activate sglang
# 启动模型部署
python -m sglang.launch_server --model-path /mnt/hwfile/llm-safety/models/huggingface/Qwen/Qwen2.5-14B-Instruct-AWQ --host 0.0.0.0 --port 10095

# 激活 rlhf 环境
conda activate rlhf
# 启动奖励模型服务
python -m openrlhf.cli.serve_rm_lxt \
    --host 0.0.0.0 \
    --port 10100 \
    --model_endpoint http://127.0.0.1:10095 \
    --batch_size 16 \
    --max_len 32768
