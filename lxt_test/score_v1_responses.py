import json
from pathlib import Path
from openai import OpenAI
import re
import random
import time
import multiprocessing as mp
from tqdm import tqdm

class ModelConfig:
    def __init__(self, base_url: str, model_name: str, api_key: str, weight: float = 1.0):
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key
        self.weight = weight
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

REFEREE_MODEL_CONFIGS = [
    ModelConfig(
        base_url="http://10.140.0.186:10005/v1",
        model_name="qwen",
        api_key="EMPTY",
        weight=1.0
    ),
    ModelConfig(
        base_url="http://10.140.0.186:10006/v1",
        model_name="qwen",
        api_key="EMPTY",
        weight=1.0
    ),
    ModelConfig(
        base_url="http://10.140.0.186:10007/v1",
        model_name="qwen",
        api_key="EMPTY",
        weight=1.0
    ),
    ModelConfig(
        base_url="http://10.140.0.186:10008/v1",
        model_name="qwen",
        api_key="EMPTY",
        weight=1.0
    ),
    ModelConfig(
        base_url="http://10.140.0.186:10009/v1",
        model_name="qwen",
        api_key="EMPTY",
        weight=1.0
    ),
    ModelConfig(
        base_url="http://10.140.0.186:10010/v1",
        model_name="qwen",
        api_key="EMPTY",
        weight=1.0
    ),
    ModelConfig(
        base_url="http://10.140.0.186:10011/v1",
        model_name="qwen",
        api_key="EMPTY",
        weight=1.0
    ),
    ModelConfig(
        base_url="http://10.140.0.186:10012/v1",
        model_name="qwen",
        api_key="EMPTY",
        weight=1.0
    )
]

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
            max_tokens=1000,
            timeout=None,
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

def process_file(input_path: str, output_path: str, retry_errors: bool = True):
    """处理单个输入文件并生成评分结果
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        retry_errors: 是否处理之前评判失败的prompt，默认为True
                     设置为True时，会处理所有之前失败的prompt
    """
    # 如果输出文件存在，加载已有结果
    existing_results = []
    existing_results_map = {}
    if Path(output_path).exists():
        try:
            with open(output_path, 'r') as f:
                existing_results = json.load(f)
                # 创建unique_id到结果的映射，用于快速查找
                existing_results_map = {item['unique_id']: item for item in existing_results if item is not None}
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {output_path}, starting fresh")
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # 初始化结果列表和需要处理的条目
    results = []
    items_to_process = []
    
    for i, item in enumerate(data):
        unique_id = item['unique_id']
        if unique_id in existing_results_map:
            results.append(existing_results_map[unique_id])
            if retry_errors:
                items_to_process.append((i, item))
        else:
            results.append(None)
            items_to_process.append((i, item))
    
    # 根据权重随机选择模型配置
    total_weight = sum(config.weight for config in REFEREE_MODEL_CONFIGS)
    rand_val = random.uniform(0, total_weight)
    cumulative_weight = 0
    for config in REFEREE_MODEL_CONFIGS:
        cumulative_weight += config.weight
        if rand_val <= cumulative_weight:
            referee_config = config
            break
    
    # 处理需要处理的条目
    for i, item in items_to_process:
        scored_responses = []
        
        for response in item['model_responses']:
            # 如果是生成错误，直接记录结果
            if response == "ERROR: Failed to generate response":
                scored_responses.append({
                    "response": response,
                    "judgment": {
                        "full_evaluation": "ERROR: Generation failed",
                        "score": 0,
                        "is_acceptable": False
                    }
                })
                continue
                
            # 如果已有结果且没有错误，保留原结果
            if (results[i] is not None and
                any(resp['response'] == response and 
                    not resp['judgment'].get('error')
                    for resp in results[i]['model_responses'])):
                scored_responses.append(next(
                    resp for resp in results[i]['model_responses']
                    if resp['response'] == response
                ))
                continue
                
            try:
                judgment = judge_response(item['prompt'], response, item['target'], referee_config)
                scored_responses.append({
                    "response": response,
                    "judgment": judgment
                })
            except Exception as e:
                print(f"Error judging response: {e}")
                scored_responses.append({
                    "response": response,
                    "judgment": {
                        "full_evaluation": "ERROR: Judging failed",
                        "score": 0,
                        "is_acceptable": False,
                        "error": str(e)
                    }
                })
        
        # 直接复制原始条目，仅替换model_responses
        result_item = item.copy()
        result_item['model_responses'] = [
            {
                "response": resp['response'],
                "judgment": resp['judgment']
            }
            for resp in scored_responses
        ]
        results[i] = result_item
        
        # 实时保存结果
        with open(output_path, 'w') as f:
            json.dump([item for item in results if item is not None], f, indent=2)
    
    return results

def process_file_wrapper(args):
    input_file, output_file, retry_errors = args
    try:
        start_time = time.time()
        process_file(str(input_file), str(output_file), retry_errors)
        duration = time.time() - start_time
        print(f"Successfully processed {input_file} -> {output_file} in {duration:.2f}s")
        return True
    except Exception as e:
        print(f"Failed to process {input_file}: {str(e)}")
        return False

def main():
    input_dir = "/mnt/petrelfs/lixiangtian/workspace/OpenRLHF/lxt_test/multi_instruct_output_10000_20241222_060400/output"
    output_dir = "/mnt/petrelfs/lixiangtian/workspace/OpenRLHF/lxt_test/multi_instruct_output_10000_20241222_060400/scored_output"
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # 获取所有输入文件
    input_files = list(Path(input_dir).glob("*.json"))
    
    # 准备输出文件路径
    output_files = [Path(output_dir) / f"scored_{input_file.name}" 
                   for input_file in input_files]
    
    # 设置是否重试错误
    retry_errors = True  # 默认重试错误项
    
    # 创建进程池，进程数等于文件数
    with mp.Pool(min(len(input_files), mp.cpu_count() * 4)) as pool:
        results = list(tqdm(
            pool.imap(process_file_wrapper, zip(input_files, output_files, [retry_errors]*len(input_files))),
            total=len(input_files),
            desc="Processing files"
        ))
        
        # 统计处理结果
        success_count = sum(results)
        failure_count = len(results) - success_count
        
        print(f"\nProcessing completed. Success: {success_count}, Failures: {failure_count}")
        
        if failure_count > 0:
            print("Failed files:")
            for input_file, success in zip(input_files, results):
                if not success:
                    print(f"- {input_file}")

if __name__ == "__main__":
    # time.sleep(1000)
    main()
