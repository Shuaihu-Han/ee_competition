import json


def load_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    records = []
    for line in lines:
        record = json.loads(line)
        records.append(record)
    return records


records = load_data('./datasets/FewFC/cascading_sampled/trainAndDev.json')

for i, item in enumerate(records):
    occur = item['occur']
    type = item['type']
    triggers = item['triggers']
    index = item['index']
    args = item['args']
    id = item['id']
    content = item['content']
    # if len(triggers) == 0:
    #     print(i)
    #     print('no triggers')
    if len(occur) == 0:
        print(i)
        print('no occur')