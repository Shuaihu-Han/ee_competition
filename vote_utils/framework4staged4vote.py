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
from vote_utils.predict_without_oracle_staged4vote_split_vector_mean import extract_all_items_without_oracle
from utils.predict_with_oracle import predict_one
from tqdm import tqdm
from utils.metric import score, gen_idx_event_dict, cal_scores, cal_scores_ti_tc_ai_ac
from utils.utils_io_data import read_jsonl, write_jsonl, cas_print


class Framework(object):

    def __init__(self, config, model_list):
        self.config = config
        self.models = [model.to(config.device) for model in model_list]

    def load_models(self, model_path_list):
        self.models = [load_model(model, model_path) for model,model_path in zip(self.models, model_path_list)]

    def evaluate_without_oracle(self, config, model_list, data_loader, device, seq_len, id_type, id_args, ty_args_id):
        for model in model_list:
            if torch.cuda.device_count() > 1 and hasattr(model, "module"):
                model = model.module
            model.eval()
        results = []
        for i, (idx, content, token, seg, mask) in tqdm(enumerate(data_loader)):
            idx = idx[0]
            token = torch.LongTensor(token).to(device)
            seg = torch.LongTensor(seg).to(device)
            mask = torch.LongTensor(mask).to(device)
            result = extract_all_items_without_oracle(model_list, device, idx, content, token, seg, mask, seq_len, config.threshold_0, config.threshold_1, config.threshold_2, config.threshold_3, config.threshold_4, id_type, id_args, ty_args_id)
            results.append(result)
        pred_records = results
        pred_dict = gen_idx_event_dict(pred_records)
        gold_records = read_jsonl(self.config.test_path)
        gold_dict = gen_idx_event_dict(gold_records)
        prf_s = cal_scores_ti_tc_ai_ac(pred_dict, gold_dict)
        return prf_s, pred_records
