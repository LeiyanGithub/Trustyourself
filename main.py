import argparse
import os
import openai
import json

from tqdm import tqdm
from transformers import GPT2Tokenizer

from contextlib import contextmanager
from collections import defaultdict

from evaluate import (
    eval_question_answering
)

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
    passage = item['passage'].replace("\"", "")
    prompt = prompt.replace('{query}', query)
    if item.get('passage'):
        prompt = prompt.replace('{passage}', passage)
    
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

# def complete(
#     prompt, max_tokens=100, temperature=0, logprobs=None, n=1,
#     frequency_penalty=0, presence_penalty=0, stop=None, rstrip=False,
#     partition_id=None, **kwargs
# ) -> str:
#     openai.api_base = "https://one.aiskt.com/v1"
#     openai.api_key = "sk-Bd2ysg0m6cQ4qVxm393aDb4720F14c10Af9cEdBf35A120Ba"
#     outputs = []
#     for message in prompt:
#         print("item: ", message)
#         completions = openai.ChatCompletion.create(
#                 model='gpt-3.5-turbo',
#                 messages=message,
#                 temperature=temperature,
#                 max_tokens=max_tokens,
#                 top_p=1,  # Not recommended to change with temperature
#         )
#         print("completions: ", completions)
#         outputs.append(completions['choices'][0]['message']['content'])
    
#     return outputs

def run_main(inlines, outfile, engine, prompt, max_tokens, n=1, temp=0, end="**"):
    if os.path.exists(outfile):
        outs = open(outfile, 'a', encoding='utf8')
        num_lines = len(open(outfile, 'r').readlines())
        inlines = inlines[num_lines - 1: ]
    else:
        outs = open(outfile, 'a', encoding='utf8')
        outs.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + '\n')
    pbar = tqdm(total = len(inlines))
    index = 0
    pbar.update(index)

    while index < len(inlines):
        inputs, answers = [], []
        inputs_with_prompts = []
        for _ in range(20):
            if index >= len(inlines): break
            input_with_prompt = add_prompt(inlines[index], prompt)
            if index == 0: 
                print(input_with_prompt)
            inputs.append(inlines[index]['question']) ## a string
            answers.append(inlines[index]['answer']) ## a list of strings only for chatbot
            inputs_with_prompts.append(input_with_prompt)
            index += 1

        outputs = complete(inputs_with_prompts, max_tokens=max_tokens, temperature=temp, n=n, stop=end)

        for i in range(len(inputs_with_prompts)):
            outs.write(json.dumps({
                'question': inputs[i], 
                'answer': answers[i], 
                'output': [outputs[i]]}, ensure_ascii=False) 
                +'\n')

        pbar.update(len(inputs_with_prompts))

    pbar.close()
    outs.close()

def qa(args, max_tokens, prompt):
    inputfile = f'./dataset/{args.dataset}.json'
    inlines = readfiles(inputfile)
    if args.nums:
        inlines = inlines[:args.nums]
    os.makedirs(args.output_dir, exist_ok=True)
    outputfile = f'{args.output_dir}/{args.model}/output.json'
    
    run_main(inlines, outputfile, args.model, prompt, max_tokens, 10, 0.7)

    evalfile = f'{args.output_dir}/{args.model}/metrics.jsonl'
    with open(evalfile, 'a', encoding='utf8') as evalout:
        emscore, coverem, length, f1 = eval_question_answering(outputfile)
        outmetrics = {
            'outputfile': outputfile,
            'prompt': prompt,
            'exact match': emscore,
            'cover_em': coverem,
            'F1': f1,
            'length': length,
            'nums':args.nums
        }
        print(f'Exact Match: {emscore}; Cover Exact Match: {coverem}; F1: {f1}; Avg.Length: {length}')
        evalout.write(json.dumps(outmetrics) + '\n')

def rewrite(args, max_tokens, prompt):
    # 需要将 passage 改写
    inputfile = f'./dataset/{args.dataset}.json'
    inlines = readfiles(inputfile)
    if args.nums:
        inlines = inlines[:args.nums]
    os.makedirs(args.output_dir, exist_ok=True)

    outputfile = f'{args.output_dir}/{args.model}/rewrite.json'
    run_main(inlines, outputfile, args.model, prompt, max_tokens, 10, 0.7)

    # evalfile = f'{args.output_dir}/{args.model}/metrics.jsonl'
    # with open(evalfile, 'a', encoding='utf8') as evalout:
    #     emscore, coverem, length, f1 = eval_question_answering(outputfile, args.endwith)
    #     outmetrics = {
    #         'outputfile': outputfile,
    #         'prompt': prompt,
    #         'exact match': emscore,
    #         'cover_em': coverem,
    #         'F1': f1,
    #         'length': length,
    #         'nums':args.nums
    #     }
    #     print(f'Exact Match: {emscore}; Cover Exact Match: {coverem}; F1: {f1}; Avg.Length: {length}')
    #     evalout.write(json.dumps(outmetrics) + '\n')

if __name__ == "__main__":

    # 涉及一个source ,target的目录和文件
    parser = argparse.ArgumentParser()
    
    # Required parameters
    parser.add_argument("--dataset", default='hotpotqa', type=str, required=True,
        help="dataset name: [nq, tqa, webq, wizard, fever, fm2]",
    )
    parser.add_argument("--model", default='llama-2-7b-chat-hf', type=str, required=False,
        help="text-davinci-002 (used in our experiments), code-davinci-002",
    )
    parser.add_argument("--task", default='question answering', type=str, required=False)
    parser.add_argument('--promptfile', default='myprompt', type=str)
    parser.add_argument('--nums', type=int, default=1000)
    parser.add_argument('--output_dir',type=str, default='./outputs')

    args = parser.parse_args()
    args.output_dir = f'./output/{args.dataset}'
    promptfile = args.promptfile

    promptlines = open(f'./prompt/{promptfile}.json', 'r').readlines()
    prompt = ''
    for line in promptlines:
        line = json.loads(line)
        if line['type'] == args.task:
            prompt = line['prompt']

    if args.task == 'question answering':
        outputs = qa(args, 100, prompt)
    elif args.task == 'rewrite':
        outputs = rewrite(args, 1024, prompt)
