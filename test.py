import json
import numpy as np
from collections import Counter


def load_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    records = []
    for line in lines:
        record = json.loads(line)
        records.append(record)
    return records


# records = load_data('./datasets/FewFC/cascading_sampled/trainAndDev.json')
records = load_data('./datasets/FewFC/data/train.json')


maxlen = 0
id = 0
lena = []
for i, item in enumerate(records):
    # occur = item['occur']
    # type = item['type']
    # triggers = item['triggers']
    # index = item['index']
    # args = item['args']
    # id = item['id']
    # content = item['content']
    # # if len(triggers) == 0:
    # #     print(i)
    # #     print('no triggers')
    # if len(occur) == 0:
    #     print(i)
    #     print('no occur')

    events = item['events']
    
    for e in events:
        length = len(e['trigger']['word'])
        lena.append(length)
        if maxlen < length:
            maxlen = length
            id = item['id']

print(maxlen)
print(id)

a = Counter(lena)
for k, v in a.items():
    print(k, v)