import json
import argparse
import random
import os
from typing import Dict, List

def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset format')
    parser.add_argument('--input', type=str, required=True, help='Input dataset path')
    parser.add_argument('--output', type=str, required=True, help='Output dataset path')
    parser.add_argument('--sample_size', type=int, default=None, 
                       help='Number of samples to extract (None for all)')
    return parser.parse_args()

def convert_format(data: List[Dict], base_dir: str) -> List[Dict]:
    converted = []
    for item in data:
        new_item = item.copy()
        
        # 处理conversations转换
        if 'conversations' in item:
            convs = item['conversations']
            if len(convs) >= 2:
                new_item['prompt'] = convs[0]['value']
                new_item['response'] = convs[1]['value']
                del new_item['conversations']
        
        # 处理image路径
        if 'image' in new_item:
            relative_path = new_item['image']
            if isinstance(relative_path, str):
                # 确保路径使用正确的分隔符
                relative_path = relative_path.replace('/', os.sep)
                # 添加data_images前缀
                absolute_path = os.path.join(base_dir, 'data_images', relative_path)
                new_item['image'] = absolute_path
        
        converted.append(new_item)
    
    return converted

def main():
    args = parse_args()
    
    # 获取输入文件的基准目录
    base_dir = os.path.dirname(os.path.abspath(args.input))
    
    # 读取输入数据
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换格式
    converted_data = convert_format(data, base_dir)
    
    # 抽样处理
    if args.sample_size is not None:
        sample_size = min(args.sample_size, len(converted_data))
        converted_data = random.sample(converted_data, sample_size)
    
    # 保存转换后的数据
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成! 共处理 {len(converted_data)} 条数据")

if __name__ == '__main__':
    main()