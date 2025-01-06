# 本文件用于抽取 /mnt/hwfile/llm-safety/datasets/LLaVA-NeXT-Data/llava_1_6.json 的一部分数据来调用大模型生成思维链回答，并用另一个大模型来提取答案，然后对比参考答案判断是否正确，将抽取的数据、模型生成的回复、提取的答案、正确性判定的结果都保存到新的json文件中。

# 加载json文件并打印前一项 
# json路径 /mnt/hwfile/llm-safety/datasets/LLaVA-NeXT-Data/llava_1_6.json

# import json

# # 打开并读取json文件
# with open('/mnt/hwfile/llm-safety/datasets/LLaVA-NeXT-Data/llava_1_6.json', 'r') as f:
#     data = json.load(f)

# # 打印前一项
# for i in range(1):
#     print(f"Item {i+1}:")
#     print(data[i])
#     print()

# 结果为
# Item 1:
# {'id': '000000033471', 'image': '000000033471.jpg', 'conversations': [{'from': 'human', 'value': '<image>\nWhat are the colors of the bus in the image?\nAnswer the question with GPT-T-COCO format.'}, {'from': 'gpt', 'value': 'The bus in the image is white and red.'}, {'from': 'human', 'value': 'What feature can be seen on the back of the bus?'}, {'from': 'gpt', 'value': 'The back of the bus features an advertisement.'}, {'from': 'human', 'value': 'Is the bus driving down the street or pulled off to the side?'}, {'from': 'gpt', 'value': 'The bus is driving down the street, which is crowded with people and other vehicles.'}]}

