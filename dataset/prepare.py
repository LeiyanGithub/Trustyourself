import os
from tqdm import tqdm
import json
import argparse
import random
from itertools import permutations
from fastchat.model import get_conversation_template

def read_data(path):
    datas = []
    if 'json' in path:
        with open(path, "r") as f:
            datas = json.load(f)
    print(len(datas))

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


# data_path = 'hotpot_dev_distractor_v1.json'

# hotpotqa_distractor = read_data(data_path)

# results = []
# for index in range(len(hotpotqa_distractor)):
#     tmp = {
#         "question": hotpotqa_distractor[index]['question'],
#         "answer": hotpotqa_distractor[index]['answer']}
    
#     gold_titles = list(set([item[0] for item in hotpotqa_distractor[index]["supporting_facts"]]))
#     all_titles = [item[0] for item in hotpotqa_distractor[index]['context']]
#     for item in hotpotqa_distractor[index]['context']:
#         if item[0] in gold_titles:
#             if 'pos' in tmp:
#                 tmp['pos'].append(' '.join(item[1]))
#             else:
#                 tmp['pos'] = [' '.join(item[1])]
#         else:
#             if 'neg' in tmp:
#                 tmp['neg'].append(' '.join(item[1]))
#             else:
#                 tmp['neg'] = [' '.join(item[1])]

#     results.append(tmp)

# write_list('hotpotqa_doc.json', results)
    
def reorganize_doc():
    path = 'llama-2-7b-chat-hf.json'
    dataset = read_data(path)
    temp_prompts = []
    tmp = {}
    target_file = 'llama-2-7b-chat-hf_rewrite.json'
    for index in range(len(dataset)):
        if dataset[index]['idx'] not in tmp:
            tmp[dataset[index]['idx']] = {
                "question": dataset[index]['question'],
                "answer": dataset[index]['answer'],
                "pos":[],
                "neg":[]
            }

        if ':\n\n' in dataset[index]['output']:
            tmp[dataset[index]['idx']][dataset[index]['label']].append(dataset[index]['output'].split(':\n\n')[1])
        else:
            tmp[dataset[index]['idx']][dataset[index]['label']].append(dataset[index]['output'])
    
    temp_prompts = [tmp[index] for index in range(len(tmp))]
    write_list(target_file, temp_prompts)
    
    return target_file

reorganize_doc()