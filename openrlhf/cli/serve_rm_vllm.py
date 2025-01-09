import argparse
import re
import json
from openai import OpenAI
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openrlhf.utils.logging_utils import init_logger
import requests
import time
from typing import Optional
from multiprocessing import Pool

logger = init_logger(__name__)

def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)
    return text

# 定义输出格式的JSON Schema
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {
            "type": "string",
            "description": "对助手回复的分析评价"
        },
        "score": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "评分（0.00-1.00）"
        }
    },
    "required": ["analysis", "score"],
    "additionalProperties": False
}

class RewardModelProxy:
    def __init__(self, args):
        self.model_host = args.model_host
        self.model_port = args.model_port
        self.model_name = args.model_name
        self.batch_size = args.batch_size
        self.max_length = args.max_len
        self.pad_token = "<pad>"
        self.eos_token = "}"
        self.num_processes = args.num_processes if hasattr(args, 'num_processes') else 4
        
        # 验证vLLM服务是否可用
        try:
            client = self._create_client()
            client.models.list()
            logger.info("Successfully connected to vLLM service")
        except Exception as e:
            logger.error(f"Failed to connect to vLLM service: {e}")
            raise

    def _create_client(self):
        """在每个进程中创建新的客户端"""
        return OpenAI(
            base_url=f"http://{self.model_host}:{self.model_port}/v1",
            api_key="dummy"
        )

    def _process_single(self, sample_tuple):
        """处理单个样本
        
        Args:
            sample_tuple: 包含(query, reference, vision_data, index)的元组
            - query: 对话文本
            - reference: 可选的参考答案
            - vision_data: 可选的图像数据
            - index: 样本索引
        """
        query, reference, vision_data, index = sample_tuple
        client = self._create_client()
        
        # 直接用正则表达式匹配对话内容
        pattern = r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>"
        matches = re.findall(pattern, query, re.DOTALL)
        
        # 提取user和assistant的内容
        dialog = {}
        for role, content in matches:
            dialog[role] = content.strip()
        
        # 构建评估文本
        eval_text = f"""你是一个专业的对话评估专家，你非常擅长评价assistant是否正确回答了问题。
对于涉及图像的问题，你会特别关注assistant的回答是否准确描述了图像内容。
你将会收到一个user和assistant的对话，可能包含:
1. user的问题/指令
2. assistant的回答
3. 参考答案(如果有)
4. 图像相关的问题(如果涉及图像)

请根据以下维度进行评估:
1. 回答的准确性和相关性:
   - 文本回答是否准确回应了问题
   - 对于图像问题，是否准确描述了图像内容
2. 回答的完整性:
   - 是否完整回答了所有问题点
   - 对于复杂问题是否有充分解释
3. 与参考答案的一致性(如果有):
   - 回答是否与参考答案在关键信息上一致
4. 语言表达:
   - 表述是否清晰专业
   - 结构是否合理

你的输出必须是一个JSON对象，包含以下字段：
{{
    "analysis": "你的分析内容（字符串）",
    "score": 0.85  // 0.00-1.00之间的分数，0.00表示完全不符合要求，1.00表示完全符合要求
}}

以下是需要评估的对话：

==========
<用户指令>
{dialog.get('user', '')}
</用户指令>

<助手回复>
{dialog.get('assistant', '')}
</助手回复>
"""

        if reference is not None:
            eval_text += f"""==========
<参考答案>
{reference}
</参考答案>
"""

        
        try:
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": eval_text}
                ],
                max_tokens=self.max_length,
                temperature=0.0,
                extra_body = {
                    "guided_json": RESPONSE_SCHEMA,
                }
            )
            
            result = json.loads(completion.choices[0].message.content)
            return index, float(result["score"])  # 返回索引和分数
            
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            return index, 0.0

    def _process_batch(self, queries, references=None, vision_data=None):
        # 准备数据
        data = []
        for i, query in enumerate(queries):
            ref = references[i] if references is not None else None
            vis = vision_data[i] if vision_data is not None else None
            data.append((query, ref, vis, i))
        
        # 为每个批次创建新的进程池
        with Pool(self.num_processes) as pool:
            # 使用进程池并行处理
            results = pool.map(self._process_single, data)
            
            # 按索引排序结果
            sorted_results = sorted(results, key=lambda x: x[0])
            # 只返回分数部分
            return [score for _, score in sorted_results]

    def get_reward(self, data):
        """处理奖励计算请求
        
        Args:
            data: 包含以下字段的字典:
                - queries: 必需，对话文本列表
                - references: 可选，参考答案列表
                - vision_data: 可选，图像数据列表
        """
        queries = data["queries"]
        references = data.get("references", None)
        vision_data = data.get("vision_data", None)
        
        if self.batch_size is None:
            batch_size = len(queries)
        else:
            batch_size = self.batch_size
            
        # 处理pad token
        for i in range(len(queries)):
            queries[i] = strip_sequence(queries[i], self.pad_token, self.eos_token) + self.eos_token
            if references is not None:
                references[i] = strip_sequence(references[i], self.pad_token, self.eos_token) + self.eos_token
                
        logger.info(f"Sample evaluation:\nQuery: {queries[0]}\nReference: {references[0] if references else 'None'}")
        
        scores = []
        # 批量处理
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:min(len(queries), i + batch_size)]
            batch_references = None
            if references is not None:
                batch_references = references[i:min(len(references), i + batch_size)]
            batch_vision = None
            if vision_data is not None:
                batch_vision = vision_data[i:min(len(vision_data), i + batch_size)]
            batch_scores = self._process_batch(batch_queries, batch_references, batch_vision)
            scores.extend(batch_scores)
            
        return scores

def create_app(args):
    app = FastAPI()
    reward_model = RewardModelProxy(args)
    
    @app.post("/get_reward")
    async def get_reward(request: Request):
        data = await request.json()
        try:
            # Add input validation
            if not data or "queries" not in data:
                return JSONResponse(
                    content={
                        "error": "Missing required field 'queries' in request body"
                    }, 
                    status_code=400
                )
            if not isinstance(data["queries"], list):
                return JSONResponse(
                    content={
                        "error": "Field 'queries' must be a list"
                    },
                    status_code=400
                )
            if not data["queries"]:
                return JSONResponse(
                    content={
                        "error": "Field 'queries' cannot be empty"
                    },
                    status_code=400
                )
            
            scores = reward_model.get_reward(data)
            logger.info(f"Scores: {scores}")
            return JSONResponse(content={"rewards": scores})
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return JSONResponse(content={"error": str(e)}, status_code=500)
        
    return app

def test_service(host: str, port: int, max_retries: int = 5, retry_interval: float = 2.0, test_vision: bool = False) -> Optional[bool]:
    """
    测试reward model服务是否正常运行
    
    Args:
        host: 服务主机地址
        port: 服务端口
        max_retries: 最大重试次数
        retry_interval: 重试间隔(秒)
    
    Returns:
        bool: 服务是否正常运行
    """
    # 基本文本测试
    test_data = {
        "queries": ["描述这张图片。"],
        "references": ["这是一张测试图片。"]
    }
    
    # 如果需要测试vision功能
    if test_vision:
        test_data["vision_data"] = [{
            "pixel_values": [0.0] * 100,  # Dummy image data
            "image_grid_thw": [1, 1, 1]
        }]
    
    url = f"http://{host}:{port}/get_reward"
    
    for i in range(max_retries):
        try:
            response = requests.post(url, json=test_data)
            if response.status_code == 200:
                result = response.json()
                if "rewards" in result and isinstance(result["rewards"], list):
                    logger.info("Service self-test passed successfully!")
                    return True
            logger.warning(f"Attempt {i+1}: Service test failed with status {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {i+1}: Connection failed - {str(e)}")
        
        if i < max_retries - 1:  # 如果不是最后一次尝试，则等待后重试
            time.sleep(retry_interval)
    
    logger.error("Service self-test failed after all retries")
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # vLLM服务参数
    parser.add_argument("--model_host", type=str, default="localhost",
                      help="vLLM server host")
    parser.add_argument("--model_port", type=int, default=20010,
                      help="vLLM server port")
    parser.add_argument("--model_name", type=str, default="qwen",
                      help="Model name to use for inference")
    
    # 服务参数
    parser.add_argument("--max_len", type=int, default=2048,
                      help="Maximum sequence length")
    parser.add_argument("--port", type=int, default=20020,
                      help="Port for the reward model service")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                      help="Host for the reward model service")
    parser.add_argument("--batch_size", type=int, default=64,
                      help="Batch size for processing requests")
    
    # 添加自测试相关参数
    # 测试相关参数
    parser.add_argument("--skip-test", action="store_true",
                      help="跳过服务自测")
    parser.add_argument("--test-vision", action="store_true", 
                      help="测试多模态功能")
    parser.add_argument("--test-retries", type=int, default=5,
                      help="Number of retries for service self-test")
    parser.add_argument("--test-interval", type=float, default=2.0,
                      help="Interval between test retries in seconds")
    
    parser.add_argument("--num_processes", type=int, default=64,
                      help="Number of processes for parallel processing")
    
    args = parser.parse_args()
    
    app = create_app(args)
    
    # 启动服务
    import uvicorn
    import threading
    
    server = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": args.host, "port": args.port}
    )
    server.daemon = True
    server.start()
    
    # 执行自测试
    if not args.skip_test:
        logger.info("Starting service self-test...")
        time.sleep(2)  # 等待服务完全启动
        if test_service(args.host, args.port, args.test_retries, args.test_interval, args.test_vision):
            logger.info("Service is running and responding correctly")
        else:
            logger.error("Service self-test failed, but continuing to run...")
    
    # 保持主线程运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Service shutting down...")
