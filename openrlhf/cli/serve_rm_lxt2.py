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
        self.client = OpenAI(
            base_url=f"http://{args.model_host}:{args.model_port}/v1",
            api_key="dummy"
        )
        self.batch_size = args.batch_size
        self.max_length = args.max_len
        self.pad_token = "<pad>"
        self.eos_token = "}"
        
        # 验证vLLM服务是否可用
        try:
            self.client.models.list()
            logger.info("Successfully connected to vLLM service")
        except Exception as e:
            logger.error(f"Failed to connect to vLLM service: {e}")
            raise
    
    def _process_batch(self, queries, responses, references=None):
        batch_scores = []
        
        for i, (query, response) in enumerate(zip(queries, responses)):
            reference = references[i] if references is not None else None
            # 构建评估文本
            eval_text = f"""请根据以下维度评估assistant的回答：
1. 回答的准确性和相关性
2. 回答的完整性
3. 最终答案的格式正确性
4. 语言表达的清晰度

你的输出必须是一个JSON对象，包含以下字段：
{{
    "analysis": "你的分析内容（字符串）",
    "score": 0.85  // 0.00-1.00之间的分数，0.00表示完全不符合要求，1.00表示完全符合要求
}}

以下是需要评估的对话：

==========
<用户指令>
{query}
</用户指令>

"""
            if reference is not None:
                eval_text += f"""==========
<参考答案>
{reference}
</参考答案>

"""
            eval_text += f"""==========
<助手回复>
{response}
</助手回复>

==========
请给出你的评估："""
            
            try:
                # 使用vLLM的OpenAI API进行评估
                completion = self.client.chat.completions.create(
                    model="qwen",
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
                score = float(result["score"])
                batch_scores.append(score)
                logger.debug(f"Evaluation result: {result}")
                
            except Exception as e:
                logger.error(f"Error in evaluation: {e}")
                batch_scores.append(0.0)
        
        return batch_scores

    def get_reward(self, data):
        queries = data["queries"]
        responses = data.get("responses", queries)  # 如果没有提供responses，使用queries
        references = data.get("references", None)  # 可选的参考答案
        
        if self.batch_size is None:
            batch_size = len(queries)
        else:
            batch_size = self.batch_size
            
        # 处理pad token
        for i in range(len(queries)):
            queries[i] = strip_sequence(queries[i], self.pad_token, self.eos_token) + self.eos_token
            responses[i] = strip_sequence(responses[i], self.pad_token, self.eos_token) + self.eos_token
            if references is not None:
                references[i] = strip_sequence(references[i], self.pad_token, self.eos_token) + self.eos_token
                
        logger.info(f"Sample evaluation:\nQuery: {queries[0]}\nResponse: {responses[0]}\nReference: {references[0] if references else 'None'}")
        
        scores = []
        # 批量处理
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:min(len(queries), i + batch_size)]
            batch_responses = responses[i:min(len(responses), i + batch_size)]
            batch_references = None
            if references is not None:
                batch_references = references[i:min(len(references), i + batch_size)]
            batch_scores = self._process_batch(batch_queries, batch_responses, batch_references)
            scores.extend(batch_scores)
            
        return scores

def create_app(args):
    app = FastAPI()
    reward_model = RewardModelProxy(args)
    
    @app.post("/get_reward")
    async def get_reward(request: Request):
        data = await request.json()
        try:
            scores = reward_model.get_reward(data)
            return JSONResponse(content={"rewards": scores})
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return JSONResponse(content={"error": str(e)}, status_code=500)
        
    return app

def test_service(host: str, port: int, max_retries: int = 5, retry_interval: float = 2.0) -> Optional[bool]:
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
    test_data = {
        "queries": ["这是一个测试问题"],
        "responses": ["这是一个测试回答"],
    }
    
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
    
    # 服务参数
    parser.add_argument("--max_len", type=int, default=2048,
                      help="Maximum sequence length")
    parser.add_argument("--port", type=int, default=20020,
                      help="Port for the reward model service")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                      help="Host for the reward model service")
    parser.add_argument("--batch_size", type=int, default=16,
                      help="Batch size for processing requests")
    
    # 添加自测试相关参数
    parser.add_argument("--skip-test", action="store_true",
                      help="Skip the service self-test")
    parser.add_argument("--test-retries", type=int, default=5,
                      help="Number of retries for service self-test")
    parser.add_argument("--test-interval", type=float, default=2.0,
                      help="Interval between test retries in seconds")
    
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
        if test_service(args.host, args.port, args.test_retries, args.test_interval):
            logger.info("Service is running and responding correctly")
        else:
            logger.error("Service self-test failed, but continuing to run...")
    
    # 保持主线程运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Service shutting down...")