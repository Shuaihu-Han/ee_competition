import json
import numpy as np
from collections import Counter
import torch


def load_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    records = []
    for line in lines:
        record = json.loads(line)
        records.append(record)
    return records


dict = torch.load("./models_save/model.bin")
# dict = torch.load("./models_save/best/most_epoch_model.bin")
for key in dict:
    print(key)


