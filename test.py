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

def write_file(path, datas):
    if 'json' in path:
        with open(path, "w", encoding='utf-8') as f:
            for _data in datas:
                print(' '.join(_data['context'][0][1]))
                temp = {"question": _data['question'], "answer": _data['answer'], "context": ' '.join(_data['context'][0][1])}
                f.write(json.dumps(temp, ensure_ascii=False))
                f.write('\n')
    f.close()

def write_list(save_path, sample_list):
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('[')
        for item in sample_list[:len(sample_list)-1]:
            temp = {"question": item['question'], "answer": item['answer'], "passage": ' '.join([' '.join(_item[1]) for _item in item['context']])}
            json_str = json.dumps(temp, ensure_ascii=False)
            f.write(json_str + ',\n')
        f.write(json.dumps(sample_list[-1], ensure_ascii=False))
        f.write(']')
    f.close()

data_path = './dataset/hotpot_dev_distractor_v1.json'
hotpotqa_distractor = read_data(data_path)

write_list('./dataset/hotpotqa.json', hotpotqa_distractor)

