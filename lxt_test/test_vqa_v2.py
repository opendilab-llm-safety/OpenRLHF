# 本文件用于抽取 /mnt/hwfile/llm-safety/datasets/VQA-v2/v2_OpenEnded_mscoco_train2014_questions.json 的一部分数据来调用大模型生成思维链回答，并用另一个大模型来提取答案，然后对比参考答案判断是否正确，将抽取的数据、模型生成的回复、提取的答案、正确性判定的结果都保存到新的json文件中。

# 加载json文件并打印前一项 

import json

# 打开并读取json文件
with open('/mnt/hwfile/llm-safety/datasets/VQA-v2/v2_OpenEnded_mscoco_train2014_questions.json', 'r') as f:
    data = json.load(f)

# 因为数据是字典格式，我们需要获取字典的键
keys = list(data.keys())
print(f"len(keys): {len(keys)}")
print(f"keys: {keys}")
print(f"type(data['questions']): {type(data['questions'])}")

# 打印前两项
for i in range(2):
    if i < len(data['questions']):
        print(f"Item {i+1}")
        print(data['questions'][i])
        print()

# 结果为



