import json
import copy



total_list = []
dict_temp = {
    "id": None,
    "event_list": []
}

event_temp = {
    "event_type": None,
    "trigger": {
        "text": None,
        "offset": []
    },
    "arguments": []
}

arg_temp = {
    "role": None,
    "text": None,
    "offset": None
}

with open("./results.json", encoding='utf-8') as f:
    while True:
        line = f.readline()
        if not line: # 到 EOF，返回空字符串，则终止循环
            break
        js = json.loads(line)
        dict_i = copy.deepcopy(dict_temp)
        dict_i["id"] = js["id"]
        js_event_list = js["events"]
        if len(js_event_list) == 0:
            print("sb")
        for j_event in js_event_list:
            event_i = copy.deepcopy(event_temp)
            event_i["event_type"] = j_event["type"]
            event_i["trigger"]["text"] = j_event["trigger"]["word"]
            event_i["trigger"]["offset"] = j_event["trigger"]["span"]
            j_arg_dict = j_event["args"]
            for i, (k, v) in enumerate(j_arg_dict.items()):
                for vi in v:
                    arg_i = copy.deepcopy(arg_temp)
                    arg_i["role"] = k
                    arg_i["text"] = vi["word"]
                    arg_i["offset"] = vi["span"]
                    event_i["arguments"].append(arg_i)
            dict_i["event_list"].append(event_i)
        total_list.append(dict_i)

# with open('./result_trans.json', "a+", encoding='utf-8') as outFile: # 用追加的方式打开要写入的文件，没有会自动创建
#     json.dump(total_list, outFile, ensure_ascii=False, indent=4)