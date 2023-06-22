import json


def read_json(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def read_jsonl(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    return data


def write_json(data, fn):
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


def write_jsonl(data, fn):
    with open(fn, 'w', encoding='utf-8') as f:
        for line in data:
            line = json.dumps(line, ensure_ascii=False)
            f.write(line + '\n')


def cas_print(s, f=None):
    print(s)
    if f:
        print(s, file=f)

def get_fmean_all(train_type, f_trigger, f_arg):
    if train_type == 'arg':
        return f_arg
    return f_trigger