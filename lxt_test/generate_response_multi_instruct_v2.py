language:lxt_test/generate_response_multi_instruct.py.py
import json
import random
import os
from typing import List, Dict, Any
import time
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from datetime import datetime

class ModelConfig:
    def __init__(self, base_url: str, model_name: str, api_key: str):
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

MODEL_CONFIGS = [
    ModelConfig(
        base_url="http://10.140.1.29:10005/v1",
        model_name="intern_qwq_v5",
        api_key="EMPTY"
    ),
    ModelConfig(
        base_url="http://10.140.1.144:10015/v1",
        model_name="intern_qwq_v5",
        api_key="EMPTY"
    ),
    ModelConfig(
        base_url="http://10.140.1.11:10025/v1",
        model_name="intern_qwq_v5",
        api_key="EMPTY"
    ),
    # ���加裁判员模型配置
    ModelConfig(
        base_url="YOUR_REFEREE_MODEL_BASE_URL",  # 请替换为您的裁判员模型地址
        model_name="YOUR_REFEREE_MODEL_NAME",  # 请替换为您的裁判员模型名称
        api_key="YOUR_REFEREE_MODEL_API_KEY"   # 请替换为您的裁判员模型 API 密钥
    ),
]

# 定义一个函数用于生成回复
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def generate_response(prompt: str, model_config: ModelConfig) -> str:
    try:
        response = model_config.client.chat.completions.create(
            model=model_config.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."},
                {"role": "user", "content": prompt}
            ],
            temperature=2,
            n=1,  # 每次只生成一个回复
            max_tokens=12800
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {e}")
        raise

# 定义一个函数用于让裁判员模型判断
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def judge_response(prompt: str, model_response: str, target: str, referee_config: ModelConfig) -> str:
    try:
        judge_prompt = f"Prompt: {prompt}\nGenerated Response: {model_response}\nReference Answer: {target}\n\nIs the generated response correct based on the reference answer? Answer 'Yes' or 'No'."
        response = referee_config.client.chat.completions.create(
            model=referee_config.model_name,
            messages=[
                {"role": "system", "content": "You are a strict judge evaluating the correctness of a generated response."},
                {"role": "user", "content": judge_prompt}
            ],
            temperature=0.1,  # 裁判员的温度可以设置低一些
            n=1,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during judging: {e}")
        return "ERROR: Judging failed"

def process_batch(args):
    batch, n_samples, process_id, output_dir, referee_config = args
    executor_configs = [config for config in MODEL_CONFIGS if config != referee_config] # 排除裁判员模型
    
    results = []
    output_path = Path(output_dir) / f"process_{process_id}_results.jsonl" # 每个进程单独保存
    
    for item in tqdm(batch, desc=f"Process {process_id}", position=process_id):
        prompt = item['prompt']
        target = item['target']
        model_responses = []
        
        for i in range(n_samples):
            executor_config = random.choice(executor_configs)
            try:
                response = generate_response(prompt, executor_config)
                
                # 让裁判员判断
                judgment = judge_response(prompt, response, target, referee_config)
                
                model_responses.append({"response": response, "judgment": judgment})
                
                # 保存当前样本的生成结果
                with open(output_path, "a") as f:
                    json.dump({"prompt": prompt, "target": target, "model_responses": model_responses}, f)
                    f.write('\n')
                
                if "yes" in judgment.lower():
                    break # 如果裁判员判断正确，则停止生成
            except Exception as e:
                print(f"Process {process_id} failed on sample: {e}")
                model_responses.append({"response": "ERROR: Failed to generate response", "judgment": "ERROR: Generation failed"})
                # 保存错误信息
                with open(output_path, "a") as f:
                    json.dump({"prompt": prompt, "target": target, "model_responses": model_responses}, f)
                    f.write('\n')
    
    return None # 不需要返回，因为已经实时保存

def generate_all_responses(sampled_data_path: str, output_dir: str):
    # 1. 加载抽样数据
    with open(sampled_data_path, 'r') as f:
        sampled_data = json.load(f)
    
    # 2. 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 3. 选择裁判员模型配置
    referee_config = None
    for config in MODEL_CONFIGS:
        if config.base_url == "YOUR_REFEREE_MODEL_BASE_URL": # 简单判断，可以根据实际情况修改
            referee_config = config
            break
    if not referee_config:
        raise ValueError("Referee model configuration not found. Please check MODEL_CONFIGS.")
    
    # 4. 多进程生成回复
    n_processes = min(mp.cpu_count()*10, 1000)
    n_samples = 100  # 设置最大生成次数
    
    # 将数据分成多个批次
    batch_size = len(sampled_data) // n_processes
    batches = [sampled_data[i:i + batch_size] for i in range(0, len(sampled_data), batch_size)]
    
    # 准备进程参数
    process_args = [(batch, n_samples, i, output_dir, referee_config) for i, batch in enumerate(batches)]
    
    # 使用进程池处理数据
    print(f"Starting generation with {n_processes} processes...")
    with mp.Pool(n_processes) as pool:
        pool.imap(process_batch, process_args)
    
    print(f"Processing complete. Results are saved in individual process files in {output_dir}")

if __name__ == "__main__":
    sampled_data_path = "multi_instruct_output_10000_20241221_002936"
    sampled_data_json_path = f"{sampled_data_path}/sampled_data.json"
    response_output_dir = f"{sampled_data_path}/output"
    
    generate_all_responses(sampled_data_json_path, response_output_dir)