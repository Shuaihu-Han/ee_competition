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


# dict = torch.load("./models_save/model.bin")
# for key in dict:
#     print(key)


# a = np.array([1,2,3])
# b = np.array([4,5,6])
# c = np.array([7,8,9])
# list = []
# list.append(a)
# list.append(b)
# list.append(c)
# ret = []
# for i in range(len(a)):
#     count = 0
#     for j in range(len(list)):
#         if list[j][i] > 5:
#             count += 1
#     if count >= 2:
#         ret.append(1)
#     else:
#         ret.append(0)
# print(ret)
# list_numpy = np.array(list)

# print(list_numpy)

# list_numpy = np.where(list_numpy > 5, 1, 0)

# print(list_numpy)
# print(list_numpy.shape)
# print(type(list_numpy))

a = '123'
b = a.split()
print(a)
print(b)




