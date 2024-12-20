import json
import random
from pathlib import Path
import datetime

def sample_dataset(input_file: str, sample_size: int, output_dir: str):
    # 1. 加载数据集
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    # 2. 随机抽样
    sampled_data = random.sample(data, sample_size)
    
    # 3. 保存抽样数据
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    sampled_output = output_dir / "sampled_data.json"
    with open(sampled_output, "w") as f:
        json.dump(sampled_data, f, indent=2)
    
    print(f"Sampled {sample_size} items saved to {sampled_output}")

if __name__ == "__main__":
    input_file = '/mnt/hwfile/llm-safety/datasets/MultiInstruct/train.jsonl'
    sample_size = 10000
    # 输出目录添加时间戳
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"multi_instruct_output_{sample_size}_{current_time}"
    
    sample_dataset(input_file, sample_size, output_dir)