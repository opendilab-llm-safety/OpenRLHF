import argparse
import re
import sglang as sgl
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)

def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)
    return text

# 这个正则表达式用于匹配一个特定格式的字符串，其中包含一个分析字段和一个分数字段。具体来说，它匹配的字符串格式如下：
# - 以{"analysis": 开头，后面跟着任意字符（.*），然后是换行符\n\n。
# - 接着是"score": 后面跟着一个数字，数字的格式为一个整数部分（[0-1]）后面跟着一个小数点（\.）和一到两位小数（[0-9]{1,2}）。匹配范围为0.00-1.99
# - 最后以一个换行符和}结束。
SCORE_REGEX = r"""\{"analysis": ".*"\n\n"score": ([0-1](\.[0-9]{1,2})?),\n\}"""

class RewardModelProxy:
    def __init__(self, args):
        # 设置sglang后端
        sgl.set_default_backend(sgl.RuntimeEndpoint(args.model_endpoint))
        self.batch_size = args.batch_size
        self.max_length = args.max_len
        
        # 保持与原代码一致的pad和eos token处理
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        
    def get_reward(self, queries):
        if self.batch_size is None:
            batch_size = len(queries)
        else:
            batch_size = self.batch_size
            
        # 处理pad token，与原代码保持一致
        for i in range(len(queries)):
            queries[i] = (
                strip_sequence(queries[i], self.pad_token, self.eos_token)
                + self.eos_token
            )
        logger.info(f"queries[0]: {queries[0]}")
        
        scores = []
        # 批量处理
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:min(len(queries), i + batch_size)]
            batch_scores = self._process_batch(batch_queries)
            scores.extend(batch_scores)
            
        return scores
    
    def _process_batch(self, texts):
        batch_scores = []
        for text in texts:
            # 使用sglang评估
            state = self._evaluate_text.run(text=text)
            result = state.text()
            
            # 提取分数
            score = float(re.search(r'"score": ([0-1](\.[0-9]{1,2})?)', result).group(1))
            batch_scores.append(score)
            
        return batch_scores
    
    @staticmethod
    @sgl.function
    def _evaluate_text(s, text: str):
        s += sgl.system("""你是一个专业的答案评估专家，你非常擅长评价assistant是否遵循了user的指令并做出了正确的回复。
你将会收到一个user和assistant的对话，请根据以下维度评估assistant的回答是否遵循了user的指令并做出了正确的回复，最后给出简要分析和你的打分:
1. 回答的准确性和相关性
2. 回答的完整性
3. 最终答案的格式正确性
4. 语言表达的清晰度

你的输出格式应当符合下列的正则表达式：
{SCORE_REGEX}

其中0.00-1.00之间的分数表示你认为assistant的回答是否遵循了user的指令并做出了正确的回复，0.00表示完全不遵循且答案错误，1.00表示完全遵循且答案正确。

请分析下面的对话，给出分析和打分。
""")
        s += f"""{text}
"""
        s += sgl.gen("score", max_tokens=1280, regex=SCORE_REGEX)

def create_app(args):
    app = FastAPI()
    reward_model = RewardModelProxy(args)
    
    @app.post("/get_reward")
    async def get_reward(request: Request):
        data = await request.json()
        queries = data.get("query")
        rewards = reward_model.get_reward(queries)
        result = {"rewards": rewards}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)
        
    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model endpoint
    parser.add_argument("--model_endpoint", type=str, default="http://10.1.96.87:30000",
                      help="SGLang model endpoint URL")
    
    # Original arguments
    parser.add_argument("--max_len", type=int, default=32768)
    parser.add_argument("--port", type=int, default=10100)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--batch_size", type=int, default=None)
    
    args = parser.parse_args()
    
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port)