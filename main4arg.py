# -*- encoding:utf -*-
from tqdm import tqdm

from torch.utils.data import DataLoader

from utils.params4arg import parse_args
from models.staged_model4args_origin import CasEE
from sklearn.metrics import *
from transformers import *
from utils.framework import Framework
from utils.data_loader import get_dict, collate_fn_dev, collate_fn_train, collate_fn_test, Data
import torch
import os
from utils.metric import gen_idx_event_dict
from utils.utils_io_data import read_jsonl, write_jsonl, cas_print, get_fmean_all
import datetime
import shutil

MODEL_CLASSES = {'bert': (BertConfig, BertModel, BertTokenizer), 'albert-zh': (AlbertConfig, AlbertModel, BertTokenizer), 'auto': (AutoConfig, AutoModel, AutoTokenizer)}

def main():
    if not os.path.exists('plm'):
        os.makedirs('plm')
    if not os.path.exists('models_save'):
        os.makedirs('models_save')
    if not os.path.exists('logs'):
        os.makedirs('logs')

    config = parse_args()
    config.type_id, config.id_type, config.args_id, config.id_args, config.ty_args, config.ty_args_id, config.args_s_id, config.args_e_id = get_dict(config.data_path)

    config.args_num = len(config.args_s_id.keys())
    config.type_num = len(config.type_id.keys())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device
    # config.model_type = 'bert'
    config.model_type = 'auto'


    config_class, model_class, tokenizer_class = MODEL_CLASSES[config.model_type]

    # config_plm = config_class.from_pretrained(config.model_name_or_path)
    # tokenizer = tokenizer_class.from_pretrained(config.model_name_or_path)
    # model_weight = model_class.from_pretrained(config.model_name_or_path)
    config_plm = config_class.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    tokenizer = tokenizer_class.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    model_weight = model_class.from_pretrained(config.model_name_or_path, trust_remote_code=True)

    config.hidden_size = config_plm.hidden_size

    model = CasEE(config, model_weight, pos_emb_size=config.rp_size, tokenizer = tokenizer)

    only_generate = False
    train_type = model.get_train_type()

    framework = Framework(config, model)
    
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    log_folder = f'./logs/{nowTime}'

    if not only_generate:
        config.do_train = True
        config.do_eval = True
        config.do_test = False
        config.generate_result = False

        os.makedirs(log_folder)
        shutil.copy('./utils/params4arg.py', log_folder)
    else:
        config.do_train = False
        config.do_eval = False
        config.do_test = False
        config.generate_result = True
    


    if config.do_train:
        train_set = Data(task='train', fn=config.data_path + '/cascading_sampled4arg/trainAndDev.json', tokenizer=tokenizer, seq_len=config.seq_length, args_s_id=config.args_s_id, args_e_id=config.args_e_id, type_id=config.type_id)
        train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn_train)
        dev_set = Data(task='eval_with_oracle', fn=config.data_path + '/cascading_sampled4arg/dev.json', tokenizer=tokenizer, seq_len=config.seq_length, args_s_id=config.args_s_id, args_e_id=config.args_e_id, type_id=config.type_id)
        dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, collate_fn=collate_fn_dev)

        with open(os.path.join(log_folder, 'log.txt'), "a+", encoding='utf-8') as logFile:
            framework.train(train_loader, dev_loader, logFile=logFile)

    if config.do_eval:
        framework.load_model(config.output_model_path_arg)
        dev_set = Data(task='eval_with_oracle', fn=config.data_path + '/cascading_sampled4arg/dev.json', tokenizer=tokenizer, seq_len=config.seq_length, args_s_id=config.args_s_id, args_e_id=config.args_e_id, type_id=config.type_id)
        dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, collate_fn=collate_fn_dev)
        c_ps, c_rs, c_fs, t_ps, t_rs, t_fs, a_ps, a_rs, a_fs = framework.evaluate_with_oracle(config, model, dev_loader, config.device, config.ty_args_id, config.id_type)

        f1_mean_all = get_fmean_all(train_type, t_fs, a_fs)
        # f1_mean_all = (t_fs + a_fs) / 2
        with open(os.path.join(log_folder, 'log.txt'), "a+", encoding='utf-8') as logFile:
            cas_print('Evaluate on all types:', logFile)
            cas_print("Type P: {:.3f}, Type R: {:.3f}, Type F: {:.3f}".format(c_ps, c_rs, c_fs), logFile)
            cas_print("Trigger P: {:.3f}, Trigger R: {:.3f}, Trigger F: {:.3f}".format(t_ps, t_rs, t_fs), logFile)
            cas_print("Args P: {:.3f}, Args R: {:.3f}, Args F: {:.3f}".format(a_ps, a_rs, a_fs), logFile)
            cas_print("F1 Mean All: {:.3f}".format(f1_mean_all), logFile)

    if config.do_test:
        if config.batch_size != 1:
            config.batch_size = 1
        framework.load_model(config.output_model_path_arg)

        config.test_path = 'datasets/FewFC/data/dev.json'
        dev_set = Data(task='eval_without_oracle', fn=config.test_path, tokenizer=tokenizer, seq_len=config.seq_length, args_s_id=config.args_s_id, args_e_id=config.args_e_id, type_id=config.type_id)
        dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, collate_fn=collate_fn_test)
        prf_s, pred_records = framework.evaluate_without_oracle(config, model, dev_loader, config.device, config.seq_length, config.id_type, config.id_args, config.ty_args_id)
        metric_names = ['TI', 'TC', 'AI', 'AC']
        with open(os.path.join(log_folder, 'log.txt'), "a+", encoding='utf-8') as logFile:
            cas_print(f"The number of testing instances:{len(dev_set)}", logFile)
            for i, prf in enumerate(prf_s):
                cas_print('{}: P:{:.1f}, R:{:.1f}, F:{:.1f}'.format(metric_names[i], prf[0] * 100, prf[1] * 100, prf[2] * 100), logFile)

    if config.generate_result:
        if config.batch_size != 1:
            config.batch_size = 1
        framework.load_model(config.output_model_path_arg)
        #config.test_path = 'datasets/FewFC/data/test.json'
        config.test_path = 'datasets/FewFC/data/dev.json'
        dev_set = Data(task='eval_without_oracle', fn=config.test_path, tokenizer=tokenizer, seq_len=config.seq_length, args_s_id=config.args_s_id, args_e_id=config.args_e_id, type_id=config.type_id)
        dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, collate_fn=collate_fn_test)
        print("The number of testing instances:", len(dev_set))
        prf_s, pred_records = framework.evaluate_without_oracle(config, model, dev_loader, config.device, config.seq_length, config.id_type, config.id_args, config.ty_args_id)
        metric_names = ['TI', 'TC', 'AI', 'AC']
        for i, prf in enumerate(prf_s):
            print('{}: P:{:.1f}, R:{:.1f}, F:{:.1f}'.format(metric_names[i], prf[0] * 100, prf[1] * 100, prf[2] * 100))

        # write_jsonl(pred_records, config.output_result_path)

if __name__ == '__main__':
    main()
