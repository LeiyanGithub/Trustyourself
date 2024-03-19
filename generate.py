import argparse
import openai
import json
import math
import time
import torch
import os
import asyncio


from tqdm import tqdm
from openai_generate import OpenAIChat
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from evaluate import eval_question_answering
from fastchat.model import load_model, get_conversation_template, add_model_args


def read_data(path):
    if 'json' in path:
        with open(path, "r", encoding='utf-8') as f:
            datas = json.load(f)
    else:
        with open(path, "r", encoding='utf-8') as f:
            datas = [_data.strip('\n') for _data in f.readlines()]

    return datas

def add_prompt(item, prompt):
    prompt = json.dumps(prompt, ensure_ascii=False)

    def rmreturn(s):
        s = s.replace('\n\n', ' ')
        s = s.replace('\n', ' ')
        return s.strip()

    query = item['question'].replace("\"", "")
    passage = item['passage'].replace("\"", "")
    if item.get('question'):
        prompt = prompt.replace('{query}', query)
    if item.get('passage'):
        prompt = prompt.replace('{passage}', passage)
    
    prompt = json.loads(prompt)
    return prompt

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

def convert_prompt(prompt):
    global tokenizer

    text = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True
        )

    return text

def construct_prompt(args):
    # 如果目标目录存在，则不需要新建
    # 选择prompt, pos_num, neg_num, 构造prompt，设置output

    promptlines = open(f'./prompt/{args.promptfile}.json', 'r').readlines()
    prompt = ''
    for line in promptlines:
        line = json.loads(line)
        if line['type'] == args.task:
            prompt = line['prompt']


    source_file = ''
    target_path = f'{args.output_dir}/{args.dataset}/{args.nums}/{args.task}'
    if args.source == 'raw':
        target_file = f'{target_path}/{args.target}_pos_{args.pos_num}_neg_{args.neg_num}.json'
    else:
        target_file = f'{target_path}/{args.source}_{args.target}_pos_{args.pos_num}_neg_{args.neg_num}.json'

    if os.path.exists(target_file):
        return target_file

    os.makedirs(target_path, exist_ok=True)
    if args.source == 'raw':
        # 原始文件，未被改写
        source_file = f'./dataset/{args.dataset}.json'
        dataset = readfiles(source_file)
        source_dataset = []
        index = 0
        if args.nums:
            while len(source_dataset) < args.nums:
                if 'neg' in dataset[index]:
                    source_dataset.append(dataset[index])
                index += 1
        
        temp_prompts = []
        for index in range(len(source_dataset)):
            # 构造promtp
            print(' '.join(source_dataset[index]['neg'][:args.neg_num]))
            source_dataset[index]['passage'] = ' '.join(source_dataset[index]['pos'][:args.pos_num])
            source_dataset[index]['passage'] += ' '.join(source_dataset[index]['neg'][:args.neg_num])
            input_with_prompt = add_prompt(source_dataset[index], prompt)
            temp_prompts.append({
                "question": source_dataset[index]['question'],
                "answer": source_dataset[index]['answer'],
                "prompt": input_with_prompt,
                "output": ''
            })
        
        # 将文件保存下来
        write_list(target_file, temp_prompts)
    else:
        target_path =  f'{args.output_dir}/{args.dataset}/{args.nums}/query_rewrite'
        source_file = f'{target_path}/{args.source}_pos_{args.pos_num}_neg_{args.neg_num}.json'
        source_dataset = read_data(source_file)
        
        print("source_file: ", source_file)
        temp_prompts = []
        for index in range(len(source_dataset)):
            print("index: ", index)
            # print(source_dataset[index])
            # if source_dataset[index]['output'].split(':'):
            #     source_dataset[index]['passage'] = max(source_dataset[index]['output'].split(':'), key=len)
            # else:
            source_dataset[index]['passage'] = source_dataset[index]['output']
            input_with_prompt = add_prompt(source_dataset[index], prompt)
            temp_prompts.append({
                "question": source_dataset[index]['question'],
                "answer": source_dataset[index]['answer'],
                "prompt": input_with_prompt,
                "output": ''
            })
        
        # 将文件保存下来
        write_list(target_file, temp_prompts)
    
    
    # 获取prompt
    

    # 构造文件
    # 读取source file文件
    
    

    return target_file

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
        if "vicuna" in args.target.lower():
            results_outputs.append(outputs[index].split("ASSISTANT:")[-1])
        elif 'chatglm' in args.target.lower():
            results_outputs.append(outputs[index].split("答：")[-1])
        elif 'llama' in args.target.lower():
            results_outputs.append(outputs[index].split("[/INST]")[-1])
        elif 'qwen' in args.target.lower():
            results_outputs.append(outputs[index].split("assistant")[-1])
    
    return results_outputs

def get_rerun_indices(datasets):
    rerun_indices_tmp = []
    for index in range(len(datasets)):
        if 'output' not in datasets[index]:
            rerun_indices_tmp.append(index)
        else:
            if datasets[index]['output'] == '':
                rerun_indices_tmp.append(index)
    
    print("rerun_indices_tmp: ", rerun_indices_tmp)
    return rerun_indices_tmp

async def run_with_batch_generate(args, dataset, save_path, batch_size=4, rerun=False, rerun_indices=[]):

    sample_list = dataset
    batch_size = args.batch_size
    rerun_elements = sample_list if not rerun else [sample_list[i] for i in rerun_indices]
    
    if args.target != 'gpt-3.5-turbo':
        prompt_elements = [convert_prompt(ele['prompt']) for ele in rerun_elements]
    else:
        prompt_elements = [ele['prompt'] for ele in rerun_elements]

    num_batches = math.ceil(len(rerun_elements) / batch_size) # 5
        
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(rerun_elements))

        messages_list = [example for example in prompt_elements[batch_start:batch_end]]

        start = time.time()
        if args.target == 'gpt-3.5-turbo':
            responses =  await chat.async_run(messages_list)
        else:
            responses =  model_generate(messages=messages_list, temperature=0.7)

        print("response:", responses)
        end = time.time()
        print("Time cost: ", (end-start) / 60)
        for j, response in enumerate(responses):
                index = batch_start + j if rerun == False else rerun_indices[batch_start + j]
                # print("response: ", response)
                sample_list[index].update({
                            'output': response})
            
        write_list(save_path, sample_list)

def readfiles(infile):
    if infile.endswith('json'): 
        lines = json.load(open(infile, 'r', encoding='utf8'))
    elif infile.endswith('jsonl'): 
        lines = open(infile, 'r', encoding='utf8').readlines()
        lines = [json.loads(l, strict=False) for l in lines]
    else:
        raise NotImplementedError
    if len(lines) == 0:
        return []
    if len(lines[0]) == 1 and lines[0].get('prompt'): 
        lines = lines[1:]
    if 'answer' in lines[0].keys() and type(lines[0]['answer']) == str:
        for l in lines:
            l['answer'] = [l['answer']]
    return lines

def add_prompt(item, prompt):
    prompt = json.dumps(prompt, ensure_ascii=False)

    def rmreturn(s):
        s = s.replace('\n\n', ' ')
        s = s.replace('\n', ' ')
        return s.strip()

    query = item['question'].replace("\"", "")
    passage = item['passage'].replace("\"", "").replace('\n', '\\n')
    prompt = prompt.replace('{query}', query)
    if item.get('passage'):
        prompt = prompt.replace('{passage}', passage)
    
    print("prompt:", prompt)
    prompt = json.loads(prompt)
    return prompt

def complete(
    prompt, max_tokens=1024, temperature=0, logprobs=None, n=1,
    frequency_penalty=0, presence_penalty=0, stop=None, rstrip=False,
    partition_id=None, **kwargs
) -> str:
    openai.api_base = "https://key.aiskt.com/v1"
    openai.api_key = "sk-Bd2ysg0m6cQ4qVxm393aDb4720F14c10Af9cEdBf35A120Ba"
    # openai.api_key = "EMPTY"
    # openai.api_base = "http://127.0.0.1:8000/v1"
    outputs = []
    for message in prompt:
        try:
            print("message:", message)
            completions = openai.ChatCompletion.create(
                            model='gpt-3.5-turbo',
                            messages=message,
                            max_tokens=max_tokens,
                            temperature=0.7,
                            top_p=1,
                        )
            print("completions: ", completions)
            outputs.append(completions['choices'][0]['message']['content'])
            # outputs.append(completions['choices'][0]['message']['content'].split(':')[1])
        except:
            outputs.append('none')
    
    return outputs



if __name__ == "__main__":

    # 涉及一个source ,target的目录和文件
    parser = argparse.ArgumentParser()
    
    # 数据集设置
    parser.add_argument("--dataset", default='hotpotqa', type=str, required=True,
        help="dataset name: [nq, tqa, webq, wizard, fever, fm2]",
    )
    # 模型设置
    parser.add_argument("--source", default='raw', type=str, required=False,
    )
    parser.add_argument("--target", default='llama-2-7b-chat-hf', type=str, required=False,
    )
    # gold passage 设置
    parser.add_argument("--pos_num", default=2, type=int, required=False,
    )
    # neg passage 设置
    parser.add_argument("--neg_num", default=0, type=int, required=False,
    )
    # 任务设置
    parser.add_argument("--task", default='question answering', type=str, required=False)
    # 模板设置
    parser.add_argument('--promptfile', default='myprompt', type=str)
    # 数量设置
    parser.add_argument('--nums', type=int, default=1000)
    # 输出文件设置
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_dir',type=str, default='./results')

    args = parser.parse_args()

    # 构造 prompt 到result file
    output_file = construct_prompt(args)
    device = "cuda"

    chat = OpenAIChat('gpt-3.5-turbo')
    args.target_model_path = f'/home/liulian/yan/pytorch_model/{args.target}'

    if args.target != 'gpt-3.5-turbo':
        model = AutoModelForCausalLM.from_pretrained(
            args.target_model_path, 
            low_cpu_mem_usage=True,
            ).to(device)

        tokenizer = AutoTokenizer.from_pretrained(
            args.target_model_path,
            use_fast=True,
            revision='main'
            )
        if 'llama' in args.target.lower():
            tokenizer.pad_token = tokenizer.eos_token
            model.config.eos_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = tokenizer.pad_token_id
        model.eval()
    dataset = read_data(output_file)
    rerun_indices = get_rerun_indices(dataset)

    while True:
        if len(rerun_indices) == 0:
            dataset = read_data(output_file)
            rerun_indices = get_rerun_indices(dataset)
            print("Processing completed.")
            if len(rerun_indices) == 0:
                print("Really end!")
                break
        else:
            print("length:  {}...\n{}".format(len(rerun_indices), rerun_indices[:10]))
        
        rerun_indices = asyncio.run(run_with_batch_generate(args, dataset, output_file, rerun=True, rerun_indices=rerun_indices))
        
        if not rerun_indices:
            rerun_indices = []
        time.sleep(30)
