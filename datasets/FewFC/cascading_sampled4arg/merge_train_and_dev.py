import json
train = []
dev = []
with open("./train.json", encoding='utf-8') as f:
    while True:
        line_train = f.readline()
        
        if not line_train: # 到 EOF，返回空字符串，则终止循环
            break
        train.append(line_train)

with open("./dev.json", encoding='utf-8') as f:
    while True:
        line_dev = f.readline()
        
        if not line_dev: # 到 EOF，返回空字符串，则终止循环
            break
        js = json.loads(line_dev)
        js['id'] = 'dev' + js['id']
        dev.append(json.dumps(js, ensure_ascii=False) + "\n")

total = train + dev
with open('./trainAndDev.json', "w+", encoding='utf-8') as outFile: # 用追加的方式打开要写入的文件，没有会自动创建
    # json.dump(total, outFile, ensure_ascii=False, indent=4)
    
    outFile.write(''.join(total))