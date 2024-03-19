# import os
# from tqdm import tqdm
# import json
# import argparse
# import random
# from itertools import permutations
# from fastchat.model import get_conversation_template

# def read_data(path):
#     datas = []
#     if 'json' in path:
#         with open(path, "r") as f:
#             datas = json.load(f)
#     print(len(datas))

#     return datas

# def write_file(path, datas):
#     if 'json' in path:
#         with open(path, "w", encoding='utf-8') as f:
#             for _data in datas:
#                 print(' '.join(_data['context'][0][1]))
#                 temp = {"question": _data['question'], "answer": _data['answer'], "context": ' '.join(_data['context'][0][1])}
#                 f.write(json.dumps(temp, ensure_ascii=False))
#                 f.write('\n')
#     f.close()

# def write_list(save_path, sample_list):
#     with open(save_path, 'w', encoding='utf-8') as f:
#         f.write('[')
#         for item in sample_list[:len(sample_list)-1]:
#             temp = {"question": item['question'], "answer": item['answer'], "passage": ' '.join([' '.join(_item[1]) for _item in item['context']])}
#             json_str = json.dumps(temp, ensure_ascii=False)
#             f.write(json_str + ',\n')
#         f.write(json.dumps(sample_list[-1], ensure_ascii=False))
#         f.write(']')
#     f.close()

# data_path = './dataset/hotpot_dev_distractor_v1.json'
# hotpotqa_distractor = read_data(data_path)

# write_list('./dataset/hotpotqa.json', hotpotqa_distractor)

# from fastchat.model import load_model, get_conversation_template


# model_path = '/home/liulian/yan/pytorch_model/Qwen1.5-7B-Chat'
# device = "cuda"
# model, tokenizer = load_model(
#         model_path,
#         device=device,
#         num_gpus=1,
#         max_gpu_memory=None,
#         load_8bit=False,
#         cpu_offloading=False,
#         revision='main',
#         debug=False,
#     )
# from transformers import AutoModelForCausalLM, AutoTokenizer
# device = "cuda" # the device to load the model onto

# model = AutoModelForCausalLM.from_pretrained(
#     "/home/liulian/yan/pytorch_model/Qwen1.5-14B-Chat",
#     torch_dtype="auto",
#     device_map="auto"
# )
# tokenizer = AutoTokenizer.from_pretrained("/home/liulian/yan/pytorch_model/Qwen1.5-14B-Chat")

# prompt = "Give me a short introduction to large language model."
# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )
# model_inputs = tokenizer([text], return_tensors="pt").to(device)

# generated_ids = model.generate(
#     model_inputs.input_ids,
#     max_new_tokens=512
# )
# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]

# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(response)


from vllm import LLM

llm = LLM(model=...)  # Name or path of your model
output = llm.generate("Hello, my name is")
print(output)
