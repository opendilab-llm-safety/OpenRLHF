# 本文件用于抽取 /mnt/hwfile/llm-safety/datasets/MultiInstruct/train.jsonl 的一部分数据来调用大模型生成思维链回答，并用另一个大模型来提取答案，然后对比参考答案判断是否正确，将抽取的数据、模型生成的回复、提取的答案、正确性判定的结果都保存到新的json文件中。

# 加载json文件并打印前一项 

import json

# 打开并读取json文件
data = []
with open('/mnt/hwfile/llm-safety/datasets/MultiInstruct/train.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

print("len(data):", len(data))

# 随机抽取并打印两项
import random

sample_indices = random.sample(range(len(data)), 2)
for i, idx in enumerate(sample_indices):
    print(f"Item {i+1} (index {idx}):")
    print(json.dumps(data[idx], indent=2))


# 结果

# len(data): 562656
# Item 1 (index 19723):
# {
#   "unique_id": "mscoco_text_type_72984",
#   "image_source": "coco2014",
#   "task_name": "text_type",
#   "image_path": "raw_datasets/MSCOCO2014/train2014/COCO_train2014_000000072984.jpg",
#   "region": [
#     "450.30405405405406 54.05063291139239 476.25 59.45569620253161"
#   ],
#   "options": [
#     "handwritten",
#     "machine printed",
#     "others"
#   ],
#   "target_txt": "machine printed",
#   "prompt": "What type is the text in 450.30405405405406 54.05063291139239 476.25 59.45569620253161? \n\n[Options]: handwritten||||machine printed||||others",
#   "target": "machine printed"
# }
# Item 2 (index 434021):
# {
#   "unique_id": "VQA_scene_recognition_10062709",
#   "task_name": "VQA_scene_recognition",
#   "image_path": "./raw_datasets/MSCOCO2014/train2014/COCO_train2014_000000275557.jpg",
#   "question": "What room is this?",
#   "target_txt": "bathroom",
#   "options": [
#     "bathroom",
#     "basket",
#     "sun",
#     "cloudy",
#     "paint",
#     "outdoor",
#     "calm",
#     "snowing"
#   ],
#   "prompt": "Given a picture of certain environment, answer the following question by select an answer from the options. \n What room is this?\n\n[Options]: bathroom||||basket||||sun||||cloudy||||paint||||outdoor||||calm||||snowing",
#   "target": "bathroom"
# }
