import json
import random
from typing import List
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
from openai import OpenAI
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
    ModelConfig(
        base_url="http://10.140.1.129:20035/v1", 
        model_name="intern_qwq_v5",
        api_key="EMPTY"
    ),
]

def generate_responses(prompt: str, model_config: ModelConfig, n_samples: int = 10) -> List[str]:
    try:
        start_time = datetime.now()
        print(f"Starting request at {start_time}")
        
        response = model_config.client.chat.completions.create(
            model=model_config.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."},
                {"role": "user", "content": prompt}
            ],
            temperature=2,
            n=n_samples,
            max_tokens=12800,
            timeout=None  # 设置为 None 表示无限等待
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"Request completed at {end_time} (duration: {duration})")
        
        return [choice.message.content for choice in response.choices]
    except Exception as e:
        print(f"Error generating responses: {e}")
        raise

def process_batch(args):
    batch, n_samples, process_id, output_dir = args
    model_config = random.choice(MODEL_CONFIGS)
    
    output_file = Path(output_dir) / f"generated_responses_process_{process_id}.json"
    
    with open(output_file, "w") as f:
        f.write("[\n")
    
    for i, item in enumerate(tqdm(batch, desc=f"Process {process_id}", position=process_id)):
        prompt = item['prompt']
        
        try:
            responses = generate_responses(prompt, model_config, n_samples)
            new_item = item.copy()
            new_item['model_responses'] = responses
            
            with open(output_file, "a") as f:
                json.dump(new_item, f, indent=2)
                if i < len(batch) - 1:
                    f.write(",\n")
                else:
                    f.write("\n")
        except Exception as e:
            print(f"Process {process_id} failed: {e}")
            new_item = item.copy()
            new_item['model_responses'] = ["ERROR: Failed to generate response"] * n_samples
            with open(output_file, "a") as f:
                json.dump(new_item, f, indent=2)
                if i < len(batch) - 1:
                    f.write(",\n")
                else:
                    f.write("\n")
    
    with open(output_file, "a") as f:
        f.write("]")
    
    return None

def generate_all_responses(sampled_data_path: str, output_dir: str):
    # 1. 加载抽样数据
    with open(sampled_data_path, 'r') as f:
        sampled_data = json.load(f)
    
    # 2. 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 3. 多进程生成回复
    n_processes = 100
    n_samples = 10
    
    # 将数据分成多个批次
    batch_size = len(sampled_data) // n_processes
    batches = [sampled_data[i:i + batch_size] for i in range(0, len(sampled_data), batch_size)]
    
    # 准备进程参数
    process_args = [(batch, n_samples, i, output_dir) for i, batch in enumerate(batches)]
    
    # 使用进程池处理数据
    print(f"Starting generation with {n_processes} processes...")
    with mp.Pool(n_processes) as pool:
        results = list(pool.imap(process_batch, process_args))
    
    # 4. 等待所有进程完成
    print(f"Processing complete. Results saved in {output_dir}")

if __name__ == "__main__":
    sampled_data_path = "multi_instruct_output_10000_20241221_002936"
    sampled_data_json_path = f"{sampled_data_path}/sampled_data.json"
    response_output_dir = f"{sampled_data_path}/output"
    
    generate_all_responses(sampled_data_json_path, response_output_dir)
