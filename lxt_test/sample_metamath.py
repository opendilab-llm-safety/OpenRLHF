import json
import random

def read_jsonl(file_path):
    """读取 JSON Lines 文件"""
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def read_json_array(file_path):
    """读取 JSON 数组文件"""
    with open(file_path, 'r') as file:
        return json.load(file)

def save_jsonl(data, file_path):
    """保存为 JSON Lines 文件"""
    with open(file_path, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')

def save_json_array(data, file_path):
    """保存为 JSON 数组文件"""
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def extract_samples(data, sample_size):
    """从数据中随机抽取样本"""
    return random.sample(data, sample_size)

# 文件路径
input_file = '/mnt/hwfile/llm-safety/datasets/MetaMathQA/MetaMathQA-395K.json'
output_file = '/mnt/hwfile/llm-safety/datasets/MetaMathQA/MetaMathQA-395K-1000.json'

# 读取数据
if input_file.endswith('.jsonl'):
    data = read_jsonl(input_file)
elif input_file.endswith('.json'):
    data = read_json_array(input_file)
else:
    raise ValueError("不支持的文件格式，请使用 .jsonl 或 .json 文件")

# 抽取样本
sample_size = 1000  # 抽取的样本数量
samples = extract_samples(data, sample_size)

# 保存样本
if output_file.endswith('.jsonl'):
    save_jsonl(samples, output_file)
elif output_file.endswith('.json'):
    save_json_array(samples, output_file)
else:
    raise ValueError("不支持的文件格式，请使用 .jsonl 或 .json 文件")

print(f"已成功抽取 {sample_size} 个样本并保存为 {output_file}")