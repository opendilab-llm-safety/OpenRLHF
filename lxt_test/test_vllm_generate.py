from vllm import LLM
import PIL


llm = LLM(model="/mnt/hwfile/llm-safety/models/huggingface/llava-hf/llava-1.5-7b-hf", tensor_parallel_size=1)

# Refer to the HuggingFace repo for the correct format to use
prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"


print("\n---------------------\n")

# Load the image using PIL.Image
image = PIL.Image.open("/mnt/hwfile/llm-safety/datasets/InfoVQA/images/10002.jpeg")

# Single prompt inference
outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": {"image": image},
})

for o in outputs:
    generated_text = o.outputs[0].text
    print(f"output: {o}")
    print(f"生成的响应：{generated_text}")


print("\n---------------------\n")

# # Batch inference
# image_1 = PIL.Image.open("/mnt/hwfile/llm-safety/datasets/InfoVQA/images/10003.jpeg")
# image_2 = PIL.Image.open("/mnt/hwfile/llm-safety/datasets/InfoVQA/images/10005.jpeg")
# outputs = llm.generate(
#     [
#         {
#             "prompt": "USER: <image>\nWhat is the content of this image?\nASSISTANT:",
#             "multi_modal_data": {"image": image_1},
#         },
#         {
#             "prompt": "USER: <image>\nWhat's the color of this image?\nASSISTANT:",
#             "multi_modal_data": {"image": image_2},
#         }
#     ]
# )

# for o in outputs:
#     generated_text = o.outputs[0].text
#     print(generated_text)