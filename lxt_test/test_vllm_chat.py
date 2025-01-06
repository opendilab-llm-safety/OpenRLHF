from typing import List, Optional, Dict, Any
from vllm import LLM
from PIL import Image
from pathlib import Path
import logging
import base64
from io import BytesIO
import json
import os

class ImageProcessor:
    """处理图像相关操作的类"""
    
    @staticmethod
    def load_image(image_path: str) -> Optional[Image.Image]:
        """安全地加载图像文件
        
        Args:
            image_path (str): 图像文件路径
            
        Returns:
            Optional[Image.Image]: 成功返回PIL.Image对象，失败返回None
        """
        try:
            return Image.open(image_path)
        except (FileNotFoundError, OSError) as e:
            logging.error(f"Failed to load image {image_path}: {str(e)}")
            return None

    @staticmethod
    def encode_image(image: Image.Image) -> str:
        """将PIL图像编码为base64字符串
        
        Args:
            image (Image.Image): PIL图像对象
            
        Returns:
            str: base64编码的字符串
        """
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

class VLMModel:
    """处理VLLM模型相关操作的类"""
    
    def __init__(self, model_path: str, tensor_parallel_size: int = 1):
        """初始化VLLM模型
        
        Args:
            model_path (str): 模型路径
            tensor_parallel_size (int): 张量并行大小，默认为4
        """
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.llm = self._load_model()
        self._is_initialized = self.llm is not None
        
    def _load_model(self) -> Optional[LLM]:
        """加载VLLM模型
        
        Returns:
            Optional[LLM]: 成功返回LLM对象，失败返回None
        """
        try:
            llm = LLM(model=self.model_path, tensor_parallel_size=self.tensor_parallel_size)
            logging.info(f"Model {self.model_path} initialized successfully")
            return llm
        except Exception as e:
            logging.error(f"Failed to load model {self.model_path}: {str(e)}")
            return None

    def cleanup(self):
        """清理模型资源"""
        if self._is_initialized:
            try:
                # vllm目前没有显式的卸载方法，依赖Python的垃圾回收
                del self.llm
                self._is_initialized = False
                logging.info("Model resources cleaned up")
            except Exception as e:
                logging.error(f"Error cleaning up model resources: {str(e)}")

    def process_single_image(self, image_path: str, prompt_template: str) -> Optional[Dict[str, Any]]:
        """处理单张图像
        
        Args:
            image_path (str): 图像文件路径
            prompt_template (str): 提示词模板
            
        Returns:
            Optional[Dict[str, Any]]: 包含原始输入和输出的字典，格式为：
                {
                    "input": {
                        "prompt": str,
                        "image": str
                    },
                    "output": str
                }
        """
        if not self.llm:
            logging.error("Model not loaded")
            return None
            
        image = ImageProcessor.load_image(image_path)
        if not image:
            return None
            
        base64_image = ImageProcessor.encode_image(image)
        
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_template},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }
        
        try:
            logging.info(f"Processing image {image_path} with prompt: {prompt_template}")
            outputs = self.llm.chat([message])
            if outputs:
                result = outputs[0].outputs[0].text
                logging.info(f"Model response: {result}")
                return {
                    "input": {
                        "prompt": prompt_template,
                        "image": base64_image[:100] + "..."  # 只显示部分base64字符串
                    },
                    "output": result
                }
            return None
        except Exception as e:
            logging.error(f"Failed to process image {image_path}: {str(e)}")
            return None


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """加载配置文件
    
    Args:
        config_path (str): 配置文件路径
        
    Returns:
        Dict[str, Any]: 配置字典
    """
    default_config = {
        "model_path": "/mnt/hwfile/llm-safety/models/huggingface/Qwen/Qwen2-VL-2B-Instruct",
        "tensor_parallel_size": 1,
        "base_prompt": "What is the content of this image?\n",
        "test_images": [
            "/mnt/hwfile/llm-safety/datasets/InfoVQA/images/10002.jpeg",
            "/mnt/hwfile/llm-safety/datasets/InfoVQA/images/10003.jpeg",
            "/mnt/hwfile/llm-safety/datasets/InfoVQA/images/10005.jpeg"
        ]
    }
    logging.info(f"Loading config from {config_path}")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return {**default_config, **json.load(f)}
        except Exception as e:
            logging.warning(f"Failed to load config file: {str(e)}")
            
    return default_config

def main():
    """主函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    config = load_config()
    
    model = VLMModel(
        model_path=config["model_path"],
        tensor_parallel_size=config["tensor_parallel_size"]
    )
    
    if not model.llm:
        return
        
    logging.info("\n-------------------------------------------\n-------------------------------------------\n")
    
    try:
        # 单图像处理示例
        image_paths = config["test_images"]
        for image_path in image_paths:
            result = model.process_single_image(image_path, config["base_prompt"])
            if result:
                logging.info(f"Image result:\n{json.dumps(result, indent=2)}")
    except Exception as e:
        logging.error(f"Error processing images: {str(e)}")
    finally:
        model.cleanup()

if __name__ == "__main__":
    main()

