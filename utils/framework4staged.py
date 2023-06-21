import os
import time
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from utils.utils_io_model import load_model, save_model
import torch
import numpy as np
from sklearn.metrics import *
from utils.predict_without_oracle_staged import extract_all_items_without_oracle
from utils.predict_with_oracle import predict_one
from tqdm import tqdm
from utils.metric import score, gen_idx_event_dict, cal_scores, cal_scores_ti_tc_ai_ac
from utils.utils_io_data import read_jsonl, write_jsonl, cas_print


class Framework(object):

    def __init__(self, config, model_trigger, model_arg):
        self.config = config
        self.model_trigger = model_trigger.to(config.device)
        self.model_arg = model_arg.to(config.device)

    def load_models(self, model_trigger_path, model_arg_path):
        self.model_trigger = load_model(self.model_trigger, model_trigger_path)
        self.model_arg = load_model(self.model_arg, model_arg_path) 

    def evaluate_without_oracle(self, config, model_trigger, model_arg, data_loader, device, seq_len, id_type, id_args, ty_args_id):
        if torch.cuda.device_count() > 1:
            model_trigger = model_trigger.module
            model_arg = model_arg.module
        model_trigger.eval()
        model_arg.eval()
        results = []
        for i, (idx, content, token, seg, mask) in tqdm(enumerate(data_loader)):
            idx = idx[0]
            token = torch.LongTensor(token).to(device)
            seg = torch.LongTensor(seg).to(device)
            mask = torch.LongTensor(mask).to(device)
            result = extract_all_items_without_oracle(model_trigger, model_arg, device, idx, content, token, seg, mask, seq_len, config.threshold_0, config.threshold_1, config.threshold_2, config.threshold_3, config.threshold_4, id_type, id_args, ty_args_id)
            results.append(result)
        pred_records = results
        pred_dict = gen_idx_event_dict(pred_records)
        gold_records = read_jsonl(self.config.test_path)
        gold_dict = gen_idx_event_dict(gold_records)
        prf_s = cal_scores_ti_tc_ai_ac(pred_dict, gold_dict)
        return prf_s, pred_records
