# 本文件用于抽取 /mnt/hwfile/llm-safety/datasets/GQA/train_balanced_questions.json 的一部分数据来调用大模型生成思维链回答，并用另一个大模型来提取答案，然后对比参考答案判断是否正确，将抽取的数据、模型生成的回复、提取的答案、正确性判定的结果都保存到新的json文件中。

# 加载json文件并打印前几项 
# json路径 /mnt/hwfile/llm-safety/datasets/GQA/train_balanced_questions.json

import json

# 打开并读取json文件
with open('/mnt/hwfile/llm-safety/datasets/GQA/train_balanced_questions.json', 'r') as f:
    data = json.load(f)

# 因为数据是字典格式，我们需要获取字典的键
keys = list(data.keys())
print(f"len(keys): {len(keys)}")

# 打印前两项
for i in range(2):
    if i < len(keys):
        key = keys[i]
        print(f"Item {i+1} (key: {key}):")
        print(data[key])
        print()

# # 结果为
# Item 1 (key: 02930152):
# {'semantic': [{'operation': 'select', 'dependencies': [], 'argument': 'sky (2486325)'}, {'operation': 'verify color', 'dependencies': [0], 'argument': 'dark'}], 'entailed': ['02930160', '02930158', '02930159', '02930154', '02930155', '02930156', '02930153'], 'equivalent': ['02930152'], 'question': 'Is the sky dark?', 'imageId': '2354786', 'isBalanced': True, 'groups': {'global': None, 'local': '06-sky_dark'}, 'answer': 'yes', 'semanticStr': 'select: sky (2486325)->verify color: dark [0]', 'annotations': {'answer': {}, 'question': {'2': '2486325'}, 'fullAnswer': {'2': '2486325'}}, 'types': {'detailed': 'verifyAttr', 'semantic': 'attr', 'structural': 'verify'}, 'fullAnswer': 'Yes, the sky is dark.'}

# Item 2 (key: 07333408):
# {'semantic': [{'operation': 'select', 'dependencies': [], 'argument': 'wall (722332)'}, {'operation': 'filter color', 'dependencies': [0], 'argument': 'white'}, {'operation': 'relate', 'dependencies': [1], 'argument': '_,on,s (722335)'}, {'operation': 'query', 'dependencies': [2], 'argument': 'name'}], 'entailed': [], 'equivalent': ['07333408'], 'question': 'What is on the white wall?', 'imageId': '2375429', 'isBalanced': True, 'groups': {'global': '', 'local': '14-wall_on,s'}, 'answer': 'pipe', 'semanticStr': 'select: wall (722332)->filter color: white [0]->relate: _,on,s (722335) [1]->query: name [2]', 'annotations': {'answer': {'0': '722335'}, 'question': {'4:6': '722332'}, 'fullAnswer': {'1': '722335', '5': '722332'}}, 'types': {'detailed': 'relS', 'semantic': 'rel', 'structural': 'query'}, 'fullAnswer': 'The pipe is on the wall.'}