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


def evaluate_result(data_file, output_file):

    with open(output_file, 'a', encoding='utf8') as evalout:
        emscore, coverem, length, f1 = eval_question_answering(data_file)
        outmetrics = {
                'outputfile': data_file,
                'exact match': emscore,
                'cover_em': coverem,
                'F1': f1,
                'length': length,
            }
        print(f'Exact Match: {emscore}; Cover Exact Match: {coverem}; F1: {f1}; Avg.Length: {length}')
        evalout.write(json.dumps(outmetrics) + '\n')


if __name__ == "__main__":

    # 涉及一个source ,target的目录和文件
    parser = argparse.ArgumentParser()
    
    # Required parameters
    parser.add_argument("--data_path", default='hotpotqa', type=str, required=True,
        help="dataset name: [nq, tqa, webq, wizard, fever, fm2]",
    )

    args = parser.parse_args()
    args.output_dir = args.data_path.replace('.json', '_metrics.json')
    evaluate_result(args.data_path, args.output_dir)
