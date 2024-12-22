import json
import random
from typing import List
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
from openai import OpenAI
from datetime import datetime
import base64
from PIL import Image
import io

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
        base_url="http://10.140.0.134:20005/v1",
        model_name="intern_qwq_v5",
        api_key="EMPTY"
    ),
    ModelConfig(
        base_url="http://10.140.1.144:20015/v1", 
        model_name="intern_qwq_v5",
        api_key="EMPTY"
    ),
    ModelConfig(
        base_url="http://10.140.1.11:20025/v1", 
        model_name="intern_qwq_v5",
        api_key="EMPTY"
    ),
    ModelConfig(
        base_url="http://10.140.1.129:20035/v1", 
        model_name="intern_qwq_v5",
        api_key="EMPTY"
    ),
]

def encode_image(image_path: str) -> str:
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if image is in RGBA format
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            # Convert image to bytes
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            image_bytes = buffer.getvalue()
            # Encode to base64
            return base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def generate_responses(item: dict, model_config: ModelConfig, n_samples: int = 10) -> List[str]:
    try:
        start_time = datetime.now()
        print(f"Starting request at {start_time}")
        
        # 读取并编码图片
        image_base64 = encode_image(item['image_path'])
        if image_base64 is None:
            raise Exception("Failed to encode image")

        # 构建消息，包含图片和提示词
        messages = [
            {
                "role": "system",
                "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": item['prompt']
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
        
        response = model_config.client.chat.completions.create(
            model=model_config.model_name,
            messages=messages,
            temperature=0.7,
            n=n_samples,
            max_tokens=25600,
            timeout=None
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
    
    # 检查是否存在已完成的文件
    completed_items = set()
    error_items = set()  # 存储需要重新处理的项
    if output_file.exists():
        try:
            # 读取不完整的JSON文件
            with open(output_file, "r") as f:
                content = f.read()
                # 如果文件不为空
                if content.strip():
                    # 确保内容是可解析的JSON数组
                    if content.startswith('['):
                        # 移除最后的逗号（如果有的话）
                        content = content.rstrip(',\n ')
                        # 添加结束括号使其成为有效的JSON
                        if not content.endswith(']'):
                            content += ']'
                        try:
                            existing_data = json.loads(content)
                            for item in existing_data:
                                # 检查是否包含错误响应
                                if any("ERROR: Failed to generate response" in resp 
                                      for resp in item.get('model_responses', [])):
                                    error_items.add(item['unique_id'])
                                else:
                                    completed_items.add(item['unique_id'])
                        except json.JSONDecodeError as e:
                            print(f"Warning: Could not parse existing file for process {process_id}: {e}")
                            # 如果解析失败，删除文件重新开始
                            output_file.unlink()
        except Exception as e:
            print(f"Error reading existing file for process {process_id}: {e}")
            output_file.unlink()
    
    # 过滤出未处理的数据和需要重新处理的数据
    remaining_batch = [item for item in batch 
                      if item.get('unique_id') not in completed_items 
                      or item.get('unique_id') in error_items]
    
    if not remaining_batch:
        print(f"Process {process_id}: All items already processed successfully")
        return None
    
    # 如果有需要重新处理的项，输出日志
    error_count = len([item for item in remaining_batch 
                      if item.get('unique_id') in error_items])
    if error_count > 0:
        print(f"Process {process_id}: Found {error_count} items with errors to retry")
    
    # 如果是新文件，写入开头的 [
    if not output_file.exists():
        with open(output_file, "w") as f:
            f.write("[\n")
    else:
        # 如果文件存在但不完整，确保它以逗号结尾
        with open(output_file, "rb+") as f:
            f.seek(0, 2)  # 移动到文件末尾
            pos = f.tell()
            if pos > 2:  # 确保文件至少有2个字节
                f.seek(pos - 2, 0)
                last_chars = f.read()
                if not last_chars.endswith(b",\n"):
                    f.seek(0, 2)  # 移动到文件末尾
                    f.write(b",\n")
    
    for i, item in enumerate(tqdm(remaining_batch, desc=f"Process {process_id}", position=process_id)):
        try:
            responses = generate_responses(item, model_config, n_samples)
            new_item = item.copy()
            new_item['model_responses'] = responses
            
            with open(output_file, "a") as f:
                json.dump(new_item, f, indent=2, ensure_ascii=False)
                if i < len(remaining_batch) - 1:
                    f.write(",\n")
                else:
                    f.write("\n")
        except Exception as e:
            print(f"Process {process_id} failed for item {item.get('unique_id', 'unknown')}: {e}")
            new_item = item.copy()
            new_item['model_responses'] = ["ERROR: Failed to generate response"] * n_samples
            with open(output_file, "a") as f:
                json.dump(new_item, f, indent=2, ensure_ascii=False)
                if i < len(remaining_batch) - 1:
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
    sampled_data_path = "/mnt/petrelfs/lixiangtian/workspace/OpenRLHF/lxt_test/multi_instruct_output_10000_20241222_060400"
    sampled_data_json_path = f"{sampled_data_path}/sampled_data.json"
    response_output_dir = f"{sampled_data_path}/output"
    
    generate_all_responses(sampled_data_json_path, response_output_dir)
