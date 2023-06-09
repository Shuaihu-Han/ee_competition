import json
import random


def load_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    records = []
    for line in lines:
        record = json.loads(line)
        records.append(record)
    return records


def write(data, fn):
    with open(fn, 'w', encoding='utf-8') as f:
        for line in data:
            line = json.dumps(line, ensure_ascii=False)
            f.write(line + '\n')


# TYPES = ['Experiment', 'Manoeuvre', 'Deploy', 'Support', 'Accident', 'Exhibit', 'Conflict', 'Injure']
TYPES = ['试验', '演习', '部署', '支援', '意外事故', '展示', '冲突', '伤亡']


def process4trigger(records):
    data_len = len(records)
    data_new = []
    for i in range(data_len):#对每个record分别处理
        record = records[i]
        data_id = record['id']
        events = record['events']
        content = record['content']

        # label all occurrence types
        type_occur = []
        for TYPE in TYPES:
            for event in events:#对每个event分别处理
                event_type = event['type']
                if event_type == TYPE:
                    type_occur.append(TYPE)#将每个event中的type全部放入type_occur中
        type_occur = list(set(type_occur))#type_occur去除重复的元素

        for TYPE in TYPES:
            if TYPE not in type_occur:
                data_dict = {}
                data_dict['id'] = data_id
                data_dict['content'] = content
                data_dict['occur'] = type_occur
                data_dict['type'] = TYPE
                data_dict['triggers'] = []
                data_dict['index'] = 0
                data_dict['args'] = []
                data_new.append(data_dict)

        # if len(type_occur) == 0:
        #     data_dict = {}
        #     data_dict['id'] = data_id
        #     data_dict['content'] = content
        #     data_dict['occur'] = type_occur
        #     data_dict['type'] = random.choice(TYPES)
        #     data_dict['triggers'] = []
        #     data_dict['index'] = 0
        #     data_dict['args'] = []
        #     data_new.append(data_dict)
        #     continue

        # label triggers and arguments
        for TYPE in TYPES:
            events_typed = []
            for event in events:#对每个event分别处理
                event_type = event['type']
                if event_type == TYPE:
                    events_typed.append(event)#event_typed中为每个TYPE对应的所有event
            # label triggers
            if len(events_typed) != 0:
                triggers = []
                trigger_args = {}
                for event in events_typed:#当前TYPE对应的event_typed中的event
                    trigger = event['trigger']['span']#event的span
                    if trigger not in triggers:
                        triggers.append(trigger)
                    trigger_args[str(trigger)] = trigger_args.get(str(trigger), {})
                    for arg_role in event['args']:
                        trigger_args[str(trigger)][arg_role] = trigger_args[str(trigger)].get(arg_role, [])
                        args_roled_spans = [item['span'] for item in event['args'][arg_role]]
                        for args_roled_span in args_roled_spans:
                            if args_roled_span not in trigger_args[str(trigger)][arg_role]:
                                trigger_args[str(trigger)][arg_role].append(args_roled_span)
                # according to trigger order, write json record
                triggers_str = [str(trigger) for trigger in triggers]  # with order
                # for trigger_str in trigger_args:
                #     index = triggers_str.index(trigger_str)
                #     data_dict = {}
                #     data_dict['id'] = data_id
                #     data_dict['content'] = content
                #     data_dict['occur'] = type_occur
                #     data_dict['type'] = TYPE
                #     data_dict['triggers'] = triggers
                #     data_dict['index'] = index
                #     data_dict['args'] = trigger_args[trigger_str]
                #     data_new.append(data_dict)

                data_dict = {}
                data_dict['id'] = data_id
                data_dict['content'] = content
                data_dict['occur'] = type_occur
                data_dict['type'] = TYPE
                data_dict['triggers'] = triggers
                data_dict['index'] = 0
                data_dict['args'] = []
                data_new.append(data_dict)
    return data_new

def process4arg(records):
    data_len = len(records)
    data_new = []
    for i in range(data_len):#对每个record分别处理
        record = records[i]
        data_id = record['id']
        events = record['events']
        content = record['content']

        # label all occurrence types
        type_occur = []
        for TYPE in TYPES:
            for event in events:#对每个event分别处理
                event_type = event['type']
                if event_type == TYPE:
                    type_occur.append(TYPE)#将每个event中的type全部放入type_occur中
        type_occur = list(set(type_occur))#type_occur去除重复的元素

        # for TYPE in TYPES:
        #     if TYPE not in type_occur:
        #         data_dict = {}
        #         data_dict['id'] = data_id
        #         data_dict['content'] = content
        #         data_dict['occur'] = type_occur
        #         data_dict['type'] = TYPE
        #         data_dict['triggers'] = []
        #         data_dict['index'] = 0
        #         data_dict['args'] = []
        #         data_new.append(data_dict)

        # if len(type_occur) == 0:
        #     data_dict = {}
        #     data_dict['id'] = data_id
        #     data_dict['content'] = content
        #     data_dict['occur'] = type_occur
        #     data_dict['type'] = random.choice(TYPES)
        #     data_dict['triggers'] = []
        #     data_dict['index'] = 0
        #     data_dict['args'] = []
        #     data_new.append(data_dict)
        #     continue

        # label triggers and arguments
        for TYPE in TYPES:
            events_typed = []
            for event in events:#对每个event分别处理
                event_type = event['type']
                if event_type == TYPE:
                    events_typed.append(event)#event_typed中为每个TYPE对应的所有event
            # label triggers
            if len(events_typed) != 0:
                triggers = []
                trigger_args = {}
                for event in events_typed:#当前TYPE对应的event_typed中的event
                    trigger = event['trigger']['span']#event的span
                    if trigger not in triggers:
                        triggers.append(trigger)
                    trigger_args[str(trigger)] = trigger_args.get(str(trigger), {})
                    for arg_role in event['args']:
                        trigger_args[str(trigger)][arg_role] = trigger_args[str(trigger)].get(arg_role, [])
                        args_roled_spans = [item['span'] for item in event['args'][arg_role]]
                        for args_roled_span in args_roled_spans:
                            if args_roled_span not in trigger_args[str(trigger)][arg_role]:
                                trigger_args[str(trigger)][arg_role].append(args_roled_span)
                # according to trigger order, write json record
                triggers_str = [str(trigger) for trigger in triggers]  # with order
                for trigger_str in trigger_args:
                    index = triggers_str.index(trigger_str)
                    data_dict = {}
                    data_dict['id'] = data_id
                    data_dict['content'] = content
                    data_dict['occur'] = type_occur
                    data_dict['type'] = TYPE
                    data_dict['triggers'] = triggers
                    data_dict['index'] = index
                    data_dict['args'] = trigger_args[trigger_str]
                    data_new.append(data_dict)
    return data_new

def generate_trigger_files():
    train = load_data('./datasets/FewFC/data/train.json')
    train = process4trigger(train)
    write(train, './datasets/FewFC/cascading_sampled4trigger/train.json')

    dev = load_data('./datasets/FewFC/data/dev.json')
    dev = process4trigger(dev)
    write(dev, './datasets/FewFC/cascading_sampled4trigger/dev.json')

    test = load_data('./datasets/FewFC/data/test.json')
    test = process4trigger(test)
    write(test, './datasets/FewFC/cascading_sampled4trigger/test.json')

def generate_arg_files():
    train = load_data('./datasets/FewFC/data/train.json')
    train = process4arg(train)
    write(train, './datasets/FewFC/cascading_sampled4arg/train.json')

    dev = load_data('./datasets/FewFC/data/dev.json')
    dev = process4arg(dev)
    write(dev, './datasets/FewFC/cascading_sampled4arg/dev.json')

    test = load_data('./datasets/FewFC/data/test.json')
    test = process4arg(test)
    write(test, './datasets/FewFC/cascading_sampled4arg/test.json')


def main():
    process_trigger_files = True
    
    if process_trigger_files:
        generate_trigger_files()
    else:
        generate_arg_files()


if __name__ == '__main__':
    main()
