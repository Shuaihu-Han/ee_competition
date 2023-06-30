import torch
import numpy as np
from collections import Counter,defaultdict
from itertools import groupby
from utils.data_loader import get_relative_pos, get_trigger_mask


def extract_all_items_without_oracle(model_list, device, idx, content: str, token, seg, mask, seq_len, threshold_0, threshold_1, threshold_2, threshold_3, threshold_4, id_type: dict, id_args: dict, ty_args_id: dict):
    assert token.size(0) == 1
    content = content[0]
    result = {'id': idx, 'content': content}
    trigger_spans = []
    trigger_spans_str = []
    events_pred = []
    candidates_pred = []
    preds = []

    typed_triggers_s = {}
    typed_triggers_e = {}
    typed_args_s = {}
    typed_args_e = {}

    for idx in range(0,len(model_list),2):
        text_emb_trigger = model_list[idx].plm(token, seg, mask)
        p_type, type_emb = model_list[idx].predict_type(text_emb_trigger, mask)
        type_pred = np.array(p_type > threshold_0, dtype=bool)
        type_pred = [i for i, t in enumerate(type_pred) if t]
        for type_pred_one in type_pred:
            type_rep = type_emb[type_pred_one, :]
            type_rep = type_rep.unsqueeze(0)
            p_s, p_e, text_rep_type = model_list[idx].predict_trigger(type_rep, text_emb_trigger, mask)

            typed_triggers_s[type_pred_one] = typed_triggers_s.get(type_pred_one, [])
            typed_triggers_s[type_pred_one].append(p_s)

            typed_triggers_e[type_pred_one] = typed_triggers_e.get(type_pred_one, [])
            typed_triggers_e[type_pred_one].append(p_e)

    for type_pred_one in type_pred:
        typed_s = np.array(typed_triggers_s.get(type_pred_one))
        typed_e = np.array(typed_triggers_e.get(type_pred_one))
        typed_s = np.where(typed_s > threshold_1, 1, 0)
        typed_e = np.where(typed_e > threshold_2, 1, 0)

        p_s = np.sum(typed_s, axis=0)
        p_e = np.sum(typed_e, axis=0)

        trigger_s = np.where(p_s > len(model_list) / 4)[0]
        trigger_e = np.where(p_e > len(model_list) / 4)[0]

        for i in trigger_s:
            es = trigger_e[trigger_e >= i]
            if len(es) > 0:
                e = es[0]
                trigger_spans.append('_'.join([str(x) for x in (type_pred_one, i, e)]))
 
    for idx in range(0,len(model_list),2):
        for k, span in enumerate(trigger_spans):
            span_str = span
            span = [int(x) for x in span.split('_')]
            text_emb_arg = model_list[idx+1].plm(token, seg, mask)
            p_type_arg, type_emb_arg = model_list[idx+1].predict_type(text_emb_arg, mask)
            type_rep_arg = type_emb_arg[span[0], :]
            type_rep_arg = type_rep_arg.unsqueeze(0)
            rp = get_relative_pos(span[1], span[2], seq_len)
            rp = [p + seq_len for p in rp]
            tm = get_trigger_mask(span[1], span[2], seq_len)
            rp = torch.LongTensor(rp).to(device)
            tm = torch.LongTensor(tm).to(device)
            rp = rp.unsqueeze(0)
            tm = tm.unsqueeze(0)
            p_s, p_e, type_soft_constrain = model_list[idx+1].predict_args(text_emb_arg, rp, tm, mask, type_rep_arg)

            p_s = np.transpose(p_s)
            p_e = np.transpose(p_e)

            typed_args_s[span_str] = typed_args_s.get(span_str, [])
            typed_args_s[span_str].append(p_s)

            typed_args_e[span_str] = typed_args_e.get(span_str, [])
            typed_args_e[span_str].append(p_e)

    for k, span in enumerate(trigger_spans):
        span_str = span
        span = [int(x) for x in span.split('_')]
        typed_s = np.array(typed_args_s.get(span_str))
        typed_e = np.array(typed_args_e.get(span_str))
        typed_s = np.where(typed_s > threshold_3, 1, 0)
        typed_e = np.where(typed_e > threshold_4, 1, 0)

        p_s = np.sum(typed_s, axis=0)
        p_e = np.sum(typed_e, axis=0)

        args_candidates = ty_args_id[span[0]]
        for i in args_candidates:
            args_s = np.where(p_s[i] > len(model_list) / 4)[0]
            args_e = np.where(p_e[i] > len(model_list) / 4)[0]
            for j in args_s:
                es = args_e[args_e >= j]
                if len(es) > 0:
                    e = es[0]
                    candidates_pred.append(str(span[0])+'_'+str(span[1])+'_'+str(span[2])+'_'+str(i)+'_'+str(j)+'_'+str(e))

    preds = [x.split('_') for x in candidates_pred]
    preds.sort(key=lambda x:(x[0],x[1],x[2]))
    for tr,args in groupby(preds, key=lambda x:(x[0],x[1],x[2])):
        one, span0, span1 = tr
        type_name = id_type[int(one)]
        pred_event_one = {'type': type_name}
        pred_trigger = {'span': [int(span0) - 1, int(span1) + 1 - 1], 'word': content[int(span0) - 1:int(span1) + 1 - 1]}  # remove <CLS> token
        pred_event_one['trigger'] = pred_trigger
        pred_args = defaultdict(list)

        for x in args:
            i, j, e = x[-3:]
            pred_arg = {'span': [int(j) - 1, int(e) + 1 - 1], 'word': content[int(j) - 1:int(e) + 1 - 1]}  # remove <CLS> token
            pred_args[id_args[int(i)]].append(pred_arg)

        # if len(pred_args) > 0:
        pred_event_one['args'] = pred_args
        events_pred.append(pred_event_one)
    result['events'] = events_pred
    return result


def extract_all_items_without_oracle_old(model_list, device, idx, content: str, token, seg, mask, seq_len, threshold_0, threshold_1, threshold_2, threshold_3, threshold_4, id_type: dict, id_args: dict, ty_args_id: dict):
    assert token.size(0) == 1
    content = content[0]
    result = {'id': idx, 'content': content}
    events_pred = []
    candidates_pred = []
    preds = []
    for idx in range(0,len(model_list),2):
        text_emb_trigger = model_list[idx].plm(token, seg, mask)
        text_emb_arg = model_list[idx+1].plm(token, seg, mask)
        p_type, type_emb = model_list[idx].predict_type(text_emb_trigger, mask)
        p_type_arg, type_emb_arg = model_list[idx+1].predict_type(text_emb_arg, mask)
        type_pred = np.array(p_type > threshold_0, dtype=bool)
        type_pred = [i for i, t in enumerate(type_pred) if t]
        for type_pred_one in type_pred:
            type_rep = type_emb[type_pred_one, :]
            type_rep = type_rep.unsqueeze(0)

            type_rep_arg = type_emb_arg[type_pred_one, :]
            type_rep_arg = type_rep_arg.unsqueeze(0)

            p_s, p_e, text_rep_type = model_list[idx].predict_trigger(type_rep, text_emb_trigger, mask)
            trigger_s = np.where(p_s > threshold_1)[0]
            trigger_e = np.where(p_e > threshold_2)[0]
            trigger_spans = []

            for i in trigger_s:
                es = trigger_e[trigger_e >= i]
                if len(es) > 0:
                    e = es[0]
                    trigger_spans.append((i, e))

            for k, span in enumerate(trigger_spans):
                rp = get_relative_pos(span[0], span[1], seq_len)
                rp = [p + seq_len for p in rp]
                tm = get_trigger_mask(span[0], span[1], seq_len)
                rp = torch.LongTensor(rp).to(device)
                tm = torch.LongTensor(tm).to(device)
                rp = rp.unsqueeze(0)
                tm = tm.unsqueeze(0)
                p_s, p_e, type_soft_constrain = model_list[idx+1].predict_args(text_emb_arg, rp, tm, mask, type_rep_arg)

                p_s = np.transpose(p_s)
                p_e = np.transpose(p_e)
                args_candidates = ty_args_id[type_pred_one]
                for i in args_candidates:
                    args_s = np.where(p_s[i] > threshold_3)[0]
                    args_e = np.where(p_e[i] > threshold_4)[0]
                    for j in args_s:
                        es = args_e[args_e >= j]
                        if len(es) > 0:
                            e = es[0]
                            candidates_pred.append(str(type_pred_one)+'_'+str(span[0])+'_'+str(span[1])+'_'+str(i)+'_'+str(j)+'_'+str(e))
    res = Counter(candidates_pred)
    for pred,cnt in res.items():
        if cnt>len(model_list)/4:
            preds.append(pred)
    preds = [x.split('_') for x in preds]
    preds.sort(key=lambda x:(x[0],x[1],x[2]))
    for tr,args in groupby(preds, key=lambda x:(x[0],x[1],x[2])):
        one, span0, span1 = tr
        type_name = id_type[int(one)]
        pred_event_one = {'type': type_name}
        pred_trigger = {'span': [int(span0) - 1, int(span1) + 1 - 1], 'word': content[int(span0) - 1:int(span1) + 1 - 1]}  # remove <CLS> token
        pred_event_one['trigger'] = pred_trigger
        pred_args = defaultdict(list)

        for x in args:
            i, j, e = x[-3:]
            pred_arg = {'span': [int(j) - 1, int(e) + 1 - 1], 'word': content[int(j) - 1:int(e) + 1 - 1]}  # remove <CLS> token
            pred_args[id_args[int(i)]].append(pred_arg)

        # if len(pred_args) > 0:
        pred_event_one['args'] = pred_args
        events_pred.append(pred_event_one)
    result['events'] = events_pred
    return result
