# add msg
import os
import re
import json
import time
import torch
import math
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from fastchat.model import load_model, get_conversation_template, add_model_args

def is_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    match = pattern.search(text)
    return match is not None

def _replace_new_line(match: re.Match[str]) -> str:
    value = match.group(2)
    value = re.sub(r"\n", r"\\n", value)
    value = re.sub(r"\r", r"\\r", value)
    value = re.sub(r"\t", r"\\t", value)
    value = re.sub('"', r"\"", value)

    return match.group(1) + value + match.group(3)

def _custom_parser(multiline_string: str) -> str:

    if isinstance(multiline_string, (bytes, bytearray)):
        multiline_string = multiline_string.decode()

    multiline_string = re.sub(
        r'("action_input"\:\s*")(.*)(")',
        _replace_new_line,
        multiline_string,
        flags=re.DOTALL,
    )
    return multiline_string


def parse_json_markdown(json_string: str) -> dict:

    match = re.search(r"```(json)?(.*)```", json_string, re.DOTALL)

    if match is None:
        json_str = json_string
    else:
        json_str = match.group(2)

    json_str = json_str.strip()

    json_str = _custom_parser(json_str)

    parsed = json.loads(json_str)

    return parsed

def read_data(path):
    if 'json' in path:
        with open(path, "r", encoding='utf-8') as f:
            datas = json.load(f)
    else:
        with open(path, "r", encoding='utf-8') as f:
            datas = [_data.strip('\n') for _data in f.readlines()]

    return datas

def write_file(path, data):
    if 'json' in path:
        with open(path, "a+", encoding="utf-8") as f:
            for _data in data:
                f.write(json.dumps(_data, ensure_ascii=False))
                f.write(',\n')
    else:

        with open(path, 'a+') as f:
            for _data in data:
                f.write(_data)
                f.write('\n')
    f.close()

def write_list(save_path, sample_list):
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('[')
        if len(sample_list) == 1:
            f.write(json.dumps(sample_list, ensure_ascii=False))
        else:
            for item in sample_list[:len(sample_list)-1]:
                json_str = json.dumps(item, ensure_ascii=False)
                f.write(json_str + ',\n')
            f.write(json.dumps(sample_list[-1], ensure_ascii=False))
        f.write(']')
    f.close()


def extract_options(text):
    pattern = re.compile(r'Options:(.*?)Answer:', re.DOTALL)
    match = pattern.search(text)

    if match:
        options_text = match.group(1)
        options = re.findall(r'([A-Z]\.\s*\w+)', options_text)

def model_generate(temperature=0.7, repetition_penalty=1.0, max_new_tokens=1024, messages=[], device = "cuda"):
    global model
    global args
    global tokenizer

    request_prompt = messages
    print("message: ", messages[0])

    tokenizer.padding_side = 'left'

    inputs = tokenizer(request_prompt, return_token_type_ids=False, padding=True, return_tensors="pt", truncation=True).to(device)
    inputs = {k: torch.tensor(v).to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens)

        results_outputs = []
        outputs = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False)

    for index in range(len(outputs)):
        if "vicuna" in args.model_path.lower():
            results_outputs.append(outputs[index].split("ASSISTANT:")[-1])
        elif 'chatglm' in args.model_path.lower():
            results_outputs.append(outputs[index].split("答：")[-1])
        elif 'llama' in args.model_path.lower():
            results_outputs.append(outputs[index].split("[/INST]")[-1])
    
    return results_outputs

def get_rerun_indices(datasets):
    rerun_indices_tmp = []
    for index in range(len(datasets)):
        if 'output' not in datasets[index]:
            rerun_indices_tmp.append(index)
        else:
            if not datasets[index]['output']:
                rerun_indices_tmp.append(index)
    
    return rerun_indices_tmp

# def extract_answer(prompt, response):
#     # 如果出现answer:，则直接抽取
#     # print("prompt: ", prompt)
#     # print("response: ", response)
#     # 如果格式不符合要求，则规则抽取
#     text = prompt
#     if "USER:" in prompt:
#         text = prompt.split("USER:")[-1].replace(' ASSISTANT:', '')
#     elif 'chatglm2' in prompt:
#         text = prompt.split("问：")[-1].replace('  答：', '')
#     elif 'llama' in prompt:
#         text = prompt.split("[INST]")[-1].replace(' [/INST]', '')

#     return text
        

def convert_prompt(model_name, prompt):
    conv = get_conversation_template(model_name)
    total_len = 0
    for message in prompt:
        if message['role'] == 'system':
            conv.system_message = message['content']
        else:
            conv.append_message(message['role'], message['content'])

    conv.append_message('assistant', None)
    prompt = conv.get_prompt()

    return prompt
    

def run_with_batch_generate(args, dataset, save_path, batch_size=2, rerun=False, rerun_indices=[]):

    sample_list = dataset
    rerun_elements = sample_list if not rerun else [sample_list[i] for i in rerun_indices]
    prompt_elements = [convert_prompt(args.model_name, ele['prompt']) for ele in rerun_elements]

    num_batches = math.ceil(len(rerun_elements) / batch_size) # 5
        
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(rerun_elements))

        # print("example['prompt']: ", rerun_elements[batch_start]['prompt'])
        messages_list = [example for example in prompt_elements[batch_start:batch_end]]
        # print("messgae: ", messages_list)

        start = time.time()
        try:
            responses =  model_generate(messages=messages_list, temperature=0.7)
        except:
            responses = []
        end = time.time()
        print("Time cost: ", (end-start) / 60)
        for j, response in enumerate(responses):
                index = batch_start + j if rerun == False else rerun_indices[batch_start + j]
                print("response: ", response)
                # extracted_answer = extract_answer(prompt_elements[index], response)
                # print("extracted_answer: ", extracted_answer)
                sample_list[index].update({
                            'output': response})
            
        write_list(save_path, sample_list)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="FastChat ChatGPT-Compatible RESTful API server."
    )
    parser.add_argument("--model_name", type=str, default="vicuna-7b")
    parser.add_argument("--model_path", type=str, default="/home/liulian/yan/pytorch_model/vicuna-7b-v1.5")
    parser.add_argument("--task", default='question answering', type=str, required=False)
    parser.add_argument("--is_role", type=str, default="")
    parser.add_argument("--is_model", type=str, default="")
    
    args = parser.parse_args()

    device = "cuda"

    if 'chatglm' in args.model_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True, revision='main'
        )
        model = AutoModel.from_pretrained(
            args.model_path, trust_remote_code=True,
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        low_cpu_mem_usage=True,
        ).to(device)

        tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        revision='main'
        )
        tokenizer.pad_token = tokenizer.eos_token

        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()

    data_path = f'./output/hotpotqa/{args.model_name}/{args.task}.json'

    dataset = read_data(data_path)
    rerun_indices = get_rerun_indices(dataset)

    while True:
        if len(rerun_indices) == 0:
            dataset = read_data(data_path)
            rerun_indices = get_rerun_indices(dataset)
            print("Processing completed.")
            if len(rerun_indices) == 0:
                print("Really end!")
                break
        else:
            print("length:  {}...\n{}".format(len(rerun_indices), rerun_indices[:10]))
        rerun_indices = run_with_batch_generate(args, dataset, data_path, rerun=True, rerun_indices=rerun_indices)
        if not rerun_indices:
            rerun_indices = []
        time.sleep(30)