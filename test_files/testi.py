import json
from abc import ABCMeta, abstractmethod
from typing import List, Dict, Union

import numpy as np
from torch import Tensor
from transformers import AutoModelForCausalLM


def read_json(json_file):
    with open(json_file, 'r') as f:
        data = [json.loads(line) for line in f]
    print(len(data))


# def upload_model():
    # model = AutoModel.from_pretrained("Alibaba-NLP/gte-Qwen2-1.5B-instruct")
    # tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-Qwen2-1.5B-instruct")


if __name__ == '__main__':
    json_file = '/cs/labs/tomhope/idopinto12/aspire_new/datasets/train/dev-cocitabsalign.jsonl'
    read_json(json_file)