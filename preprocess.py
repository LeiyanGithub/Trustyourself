import os
import json
import argparse
import random

from tqdm import tqdm
from itertools import permutations

# 构造 prompt
def add_prompt(item, prompt):

    prompt = json.dumps(prompt, ensure_ascii=False)

    def rmreturn(s):
        s = s.replace('\n\n', ' ')
        s = s.replace('\n', ' ')
        return s.strip()

    query = item['question'].replace("\"", "")
    prompt = prompt.replace('{query}', query)
    if item.get('passage'):
        passage = item['passage'].replace("\"", "")
        passage = passage.replace('\\', '\\\\')
        prompt = prompt.replace('{passage}', passage)
    
    print("prompt: ", prompt)

    prompt = json.loads(prompt)

    return prompt

def read_data(path):
    if 'json' in path:
        with open(path, "r") as f:
            datas = json.load(f)
    elif 'chatglm2' in path:
        with open(path, "r") as f:
            datas = [_data.replace("\\n", "\n").replace("\\\t", "\\t")  for _data in f.readlines()]
    else:
        with open(path, "r") as f:
            datas = [_data.rstrip('\n').replace("\\n", "\n").replace("\\t", "\t") for _data in f.readlines()]

    return datas

def write_file(path, data):
    if 'json' in path:
        with open(path, "a+", encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False))
                f.write('\n')
    else:
        with open(path, 'w', encoding='utf-8') as f:
            for _data in data:
                f.write(_data)
                f.write('\n')
    f.close()

def write_list(save_path, sample_list):
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('[')
        for item in sample_list[:len(sample_list)-1]:
            json_str = json.dumps(item, ensure_ascii=False)
            f.write(json_str + ',\n')
        f.write(json.dumps(sample_list[-1], ensure_ascii=False))
        f.write(']')
    f.close()


def get_model_answer_few_shot(description, label, question, examples):
    # 顺序打乱
    cases = "\\n ".join(list(random.choice(list(permutations(examples, 4)))))
    prompt = "Now we need you to role-play a person and to answer the questions, assuming you are {description}. ASSISTANT: ok. Now I am the person. USER: Please answer the following questions according to person's personality. Provide the final answer A or B directly, without any nonsense. ASSISTANT: ok. \\n {examples}. USER: ".format(description=description, examples=cases) + question + "\\n"
    save_path = './prompt/{label}.jsonl'.format(label=label)
    write_file(save_path, {"prompt": prompt, "gold": label})
    return None


def get_model_answer_few_shot_chatglm(description, label, question, examples):
    # 顺序打乱
    index = 3
    cases = ''
    for example in list(random.choice(list(permutations(examples, 4)))):
        cases += "[Round {index}]\n\n问：".format(index=index) + example + '\n'
        index += 1
    prompt = "[Round 1]\n\n问：Now we need you to role-play a person and to answer the questions, assuming you are {description}. \n\n答：ok. Now I am the person. \n\n[Round 2]\n\n问：Please answer the following questions according to person's personality. Provide the final answer A or B directly, without any nonsense. \n\n答：ok. \n\n{examples}".format(description=description, examples=cases) + "[Round 7]\n\n问：" + question + "\n\n答："
    save_path = './prompt/{label}.jsonl'.format(label=label)
    write_file(save_path, {"prompt": prompt, "gold": label})
    return None


def get_model_answer_zero_shot(description, label, question, examples):
    # 顺序打乱
    cases = "\\n ".join(list(random.choice(list(permutations(examples, 4)))))
    prompt = "Now we need you to role-play a person and to answer the questions, assuming you are {description},\\n Provide the final answer A or B directly, without any nonsense. Please answer the following questions according to person's personality. \\n Question: ".format(description=description) + question + "\\n Answer:"
    save_path = './prompt/{label}.jsonl'.format(label=label)
    write_file(save_path, {"prompt": prompt, "gold": label})
    return None


def get_model_answer(description, label, question, examples):
    # 顺序打乱
    cases = "\\n ".join(list(random.choice(list(permutations(examples, 4)))))
    prompt = "Now we need you to role-play a person and to answer the questions, assuming you are {description},\\n Provide the final answer A or B directly, without any nonsense. Provide a few format examples: \\n {examples}. Please answer the following questions according to person's personality. \\n Question: ".format(description=description, examples=cases) + question + "\\n Answer:"
    save_path = './prompt/{label}.jsonl'.format(label=label)
    write_file(save_path, {"prompt": prompt, "gold": label})
    return None


def get_role_mbti(description, label, mbti_questions):
    examples = read_data('./data/example_chatglm2.txt')
    for question in tqdm(list(mbti_questions.values())):
        get_model_answer_few_shot_chatglm(
            description,
            label,
            question['question'],
            examples
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default='hotpotqa', type=str, required=True,
        help="dataset name: [nq, tqa, webq, wizard, fever, fm2]",
    )
    parser.add_argument("--model", default='llama-2-7b-chat-hf', type=str, required=False,
        help="text-davinci-002 (used in our experiments), code-davinci-002",
    )
    parser.add_argument("--task", default='qa', type=str, required=False,
        help="text-davinci-002 (used in our experiments), code-davinci-002",
    )
    parser.add_argument('--promptfile', default='myprompt', type=str)
    parser.add_argument('--nums', type=int, default=1000)
    parser.add_argument('--output_dir',type=str, default='./outputs')

    hotpotqa = read_data('/home/liulian/yan/Hotpotqa/trustyourself/dataset/hotpotqa.json')

    args = parser.parse_args()
    args.output_dir = f'./output/{args.dataset}'

    promptfile = args.promptfile
    promptlines = open(f'./prompt/{promptfile}.json', 'r').readlines()

    for line in promptlines:
        line = json.loads(line)
        if line['type'] == args.task:
            prompt = line['prompt']

    outputfile = f'{args.output_dir}/{args.model}/{args.task}.json'

    if os.path.exists(outputfile):
        outs = open(outputfile, 'a', encoding='utf8')
        num_lines = len(open(outputfile, 'r').readlines())
    else:
        outs = open(outputfile, 'a', encoding='utf8')
    
    # print("prompt: ", prompt)

    prompts = []
    for index in range(1000):
        input_with_prompt = add_prompt(hotpotqa[index], prompt)
        temp = {
                'question': hotpotqa[index]['question'], 
                'answer': hotpotqa[index]['answer'], 
                'prompt': input_with_prompt,
                'output': ''}
        prompts.append(temp)
    
    write_list(outputfile, prompts)

