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
import base64
from PIL import Image
import io
import re

class ModelConfig:
    def __init__(self, base_url: str, model_name: str, api_key: str):
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

def encode_image(image_path: str) -> str:
    try:
        with Image.open(image_path) as img:
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            image_bytes = buffer.getvalue()
            return base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

# 分离执行模型和裁判员模型配置
EXECUTOR_MODEL_CONFIGS = [
    ModelConfig(
        base_url="http://10.140.0.134:10005/v1",
        model_name="intern_qwq_v5",
        api_key="EMPTY"
    ),
]

REFEREE_MODEL_CONFIGS = [
    ModelConfig(
        base_url="http://10.140.1.29:10005/v1",
        model_name="intern_qwq_v5",
        api_key="EMPTY"
    ),
]

# 定义一个函数用于生成回复
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def generate_response(item: dict, model_config: ModelConfig) -> str:
    try:
        # 读取并编码图片
        image_base64 = encode_image(item['image_path'])
        if image_base64 is None:
            raise Exception("Failed to encode image")

        response = model_config.client.chat.completions.create(
            model=model_config.model_name,
            messages=[
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
            ],
            temperature=0.7,
            n=1,
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
        judge_prompt = f"""As an AI evaluation expert, please first extract the key answer from the generated response, then assess it against the reference answer.

=== Input Information ===
<prompt>
{prompt}
</prompt>

<generated_response>
{model_response}
</generated_response>

<reference_answer>
{target}
</reference_answer>

=== Task Steps ===
1. Extract the key answer from the generated response
2. Compare the extracted answer with the reference answer
3. Evaluate based on the following criteria

=== Evaluation Criteria ===
1. Semantic Accuracy: Check if the core meaning matches
2. Numerical Precision: For numerical answers, verify approximate equality
3. Key Points Coverage: For descriptive answers, verify essential points
4. Overall Coherence: Evaluate response clarity and relevance

=== Output Format Requirements ===
Your evaluation must be provided in XML format as follows:
<evaluation>
    <extracted_answer>
        [The key answer extracted from generated response]
    </extracted_answer>
    <analysis>
        [2-3 sentences explaining the comparison]
    </analysis>
    <detailed_scores>
        <semantic_accuracy>[0-100]</semantic_accuracy>
        <numerical_precision>[0-100]</numerical_precision>
        <key_points>[0-100]</key_points>
        <coherence>[0-100]</coherence>
    </detailed_scores>
    <final_score>[0-100]</final_score>
</evaluation>

Example Output:
<evaluation>
    <extracted_answer>
        The main character's name is John.
    </extracted_answer>
    <analysis>
        The response accurately captures the main concept. The explanation is clear and well-structured.
    </analysis>
    <detailed_scores>
        <semantic_accuracy>90</semantic_accuracy>
        <numerical_precision>95</numerical_precision>
        <key_points>85</key_points>
        <coherence>88</coherence>
    </detailed_scores>
    <final_score>89</final_score>
</evaluation>

Your Evaluation:
"""

        response = referee_config.client.chat.completions.create(
            model=referee_config.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict judge evaluating the correctness of responses. You must follow the XML output format exactly."
                },
                {
                    "role": "user",
                    "content": judge_prompt
                }
            ],
            temperature=0.1,
            n=1,
            max_tokens=800,
            extra_body={
                "guided_regex": r"<evaluation>\s*<extracted_answer>[\s\S]*?</extracted_answer>\s*<analysis>[\s\S]*?</analysis>\s*<detailed_scores>\s*<semantic_accuracy>\d{1,3}</semantic_accuracy>\s*<numerical_precision>\d{1,3}</numerical_precision>\s*<key_points>\d{1,3}</key_points>\s*<coherence>\d{1,3}</coherence>\s*</detailed_scores>\s*<final_score>\d{1,3}</final_score>\s*</evaluation>"
            }
        )
        
        full_response = response.choices[0].message.content.strip()
        
        # 使用正则表达式提取评估结果
        eval_match = re.search(r'<evaluation>(.*?)</evaluation>', full_response, re.DOTALL)
        if not eval_match:
            raise ValueError("Failed to extract evaluation XML")
            
        # 提取各个部分
        extracted_answer_match = re.search(r'<extracted_answer>(.*?)</extracted_answer>', full_response, re.DOTALL)
        analysis_match = re.search(r'<analysis>(.*?)</analysis>', full_response, re.DOTALL)
        final_score_match = re.search(r'<final_score>(\d{1,3})</final_score>', full_response)
        
        # 提取详细分数
        detailed_scores = {
            'semantic_accuracy': int(re.search(r'<semantic_accuracy>(\d{1,3})</semantic_accuracy>', full_response).group(1)),
            'numerical_precision': int(re.search(r'<numerical_precision>(\d{1,3})</numerical_precision>', full_response).group(1)),
            'key_points': int(re.search(r'<key_points>(\d{1,3})</key_points>', full_response).group(1)),
            'coherence': int(re.search(r'<coherence>(\d{1,3})</coherence>', full_response).group(1))
        }

        if final_score_match:
            final_score = int(final_score_match.group(1))
            return {
                "full_evaluation": full_response,
                "extracted_answer": extracted_answer_match.group(1).strip() if extracted_answer_match else "",
                "analysis": analysis_match.group(1).strip() if analysis_match else "",
                "detailed_scores": detailed_scores,
                "final_score": final_score,
                "is_acceptable": final_score >= 80
            }
        else:
            return {
                "full_evaluation": full_response,
                "score": 0,
                "is_acceptable": False,
                "error": "Failed to extract final score"
            }
    except Exception as e:
        print(f"Error during judging: {e}")
        return {
            "full_evaluation": "ERROR: Judging failed",
            "score": 0,
            "is_acceptable": False,
            "error": str(e)
        }

def process_batch(args):
    batch, n_samples, process_id, output_dir = args
    referee_config = random.choice(REFEREE_MODEL_CONFIGS)
    
    output_path = Path(output_dir) / f"process_{process_id}_results.jsonl"
    
    for item in tqdm(batch, desc=f"Process {process_id}", position=process_id):
        model_responses = []
        
        for i in range(n_samples):
            executor_config = random.choice(EXECUTOR_MODEL_CONFIGS)
            try:
                response = generate_response(item, executor_config)
                judgment = judge_response(item['prompt'], response, item['target'], referee_config)
                
                model_responses.append({
                    "response": response,
                    "judgment": judgment,
                    "model_name": executor_config.model_name
                })
                
                # 实时保存结果
                with open(output_path, "a") as f:
                    json.dump({
                        "prompt": item['prompt'],
                        "target": item['target'],
                        "image_path": item['image_path'],
                        "model_responses": model_responses
                    }, f)
                    f.write('\n')
                
                if judgment.get("is_acceptable", False):
                    break  # 如果得分达到及格线，则停止生成
            except Exception as e:
                print(f"Process {process_id} failed: {e}")
                model_responses.append({
                    "response": "ERROR: Failed to generate response",
                    "judgment": {
                        "full_evaluation": "ERROR: Generation failed",
                        "score": 0,
                        "is_acceptable": False,
                        "error": str(e)
                    },
                    "model_name": executor_config.model_name
                })
                
                with open(output_path, "a") as f:
                    json.dump({
                        "prompt": item['prompt'],
                        "target": item['target'],
                        "image_path": item['image_path'],
                        "model_responses": model_responses,
                        "error": str(e)
                    }, f)
                    f.write('\n')
    
    return None

def generate_all_responses(sampled_data_path: str, output_dir: str):
    # 1. 加载抽样数据
    with open(sampled_data_path, 'r') as f:
        sampled_data = json.load(f)
    
    # 2. 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 3. 多进程生成回复
    n_processes = min(mp.cpu_count()*10, 1000)
    n_samples = 100  # 设置最大生成次数
    
    # 将数据分成多个批次
    batch_size = len(sampled_data) // n_processes
    batches = [sampled_data[i:i + batch_size] for i in range(0, len(sampled_data), batch_size)]
    
    # 准备进程参数
    process_args = [(batch, n_samples, i, output_dir) for i, batch in enumerate(batches)]
    
    # 使用进程池处理数据
    print(f"Starting generation with {n_processes} processes...")
    with mp.Pool(n_processes) as pool:
        pool.imap(process_batch, process_args)
    
    print(f"Processing complete. Results are saved in individual process files in {output_dir}")

if __name__ == "__main__":
    sampled_data_path = "/mnt/petrelfs/lixiangtian/workspace/OpenRLHF/lxt_test/multi_instruct_output_10000_20241222_060400"
    sampled_data_json_path = f"{sampled_data_path}/sampled_data.json"
    response_output_dir = f"{sampled_data_path}/output"
    
    generate_all_responses(sampled_data_json_path, response_output_dir)
