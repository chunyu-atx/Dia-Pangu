import copy
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from models.bert_labeler import bert_labeler
from bert_tokenizer import tokenize
from sklearn.metrics import f1_score, confusion_matrix
from statsmodels.stats.inter_rater import cohens_kappa
from transformers import BertTokenizer
from constants import *


# --- 修改 get_weighted_f1_weights 函数 ---
def get_weighted_f1_weights(train_path_or_csv):
    """
    根据原始CTRG_finding_labels.csv的格式计算F1权重。
    原始标签: 1=Positive, 2=Negative, 0=Blank
    Chexbert内部标签: 1=Positive, 2=Negative, 0=Blank, 3=Uncertain
    """
    if isinstance(train_path_or_csv, str):
        df = pd.read_csv(train_path_or_csv)
    else:
        df = train_path_or_csv

        # 这个函数用于计算加权F1，基于最终模型看到的标签分布
    # 我们模拟一下最终的标签
    # CSV '2' (neg) -> 0 -> 2
    # CSV '1' (pos) -> 1 -> 1
    # CSV '0' (blank) -> NaN -> 0
    # 所以，原始CSV中的 '2' 对应最终的 'Negative'，'1' 对应 'Positive', '0' 对应 'Blank'

    weight_dict = {}
    for cond in CONDITIONS:
        weights = []
        # 确保df中有这个列
        if cond not in df.columns:
            # 如果CSV列名是小写下划线，需要转换
            cond_csv_name = cond.replace(' ', '_').lower()
            if cond_csv_name not in df.columns:
                # 如果还是找不到，跳过
                continue
            col = df[cond_csv_name]
        else:
            col = df[cond]

        # 计算 Negation (对应原始标签 2)
        weights.append((col == 2).sum())

        # 计算 Uncertain (数据中没有, 为0)
        weights.append(0)

        # 计算 Positive (对应原始标签 1)
        weights.append((col == 1).sum())

        if np.sum(weights) > 0:
            weights = np.array(weights) / np.sum(weights)
        else:
            weights = np.array([0.33, 0.33, 0.34])  # 默认值
        weight_dict[cond] = weights
    return weight_dict

def weighted_avg(scores, weights):
    """Compute weighted average of scores
    @param scores(List): the task scores
    @param weights (List): corresponding normalized weights

    @return (float): the weighted average of task scores
    """
    return np.sum(np.array(scores) * np.array(weights))

def compute_train_weights(train_path):
    """Compute class weights for rebalancing rare classes
    @param train_path (str): A path to the training csv file

    @returns weight_arr (torch.Tensor): Tensor of shape (train_set_size), containing
                                        the weight assigned to each training example 
    """
    df = pd.read_csv(train_path)
    cond_weights = {}
    for cond in CONDITIONS:
        col = df[cond]
        val_counts = col.value_counts()
        if cond != 'No Finding':
            weights = {}
            weights['0.0'] = len(df) / val_counts[0]
            weights['-1.0'] = len(df) / val_counts[-1]
            weights['1.0'] = len(df) / val_counts[1]
            weights['nan'] = len(df) / (len(df) - val_counts.sum())
        else:
            weights = {}
            weights['1.0'] = len(df) / val_counts[1]
            weights['nan'] = len(df) / (len(df) - val_counts.sum())
            
        cond_weights[cond] = weights
        
    weight_arr = torch.zeros(len(df))
    for i in range(len(df)):     #loop over training set
        for cond in CONDITIONS:  #loop over all conditions
            label = str(df[cond].iloc[i])
            weight_arr[i] += cond_weights[cond][label] #add weight for given class' label
        
    return weight_arr

def generate_attention_masks(batch, source_lengths, device):
    """Generate masks for padded batches to avoid self-attention over pad tokens
    @param batch (Tensor): tensor of token indices of shape (batch_size, max_len)
                           where max_len is length of longest sequence in the batch
    @param source_lengths (List[Int]): List of actual lengths for each of the
                           sequences in the batch
    @param device (torch.device): device on which data should be

    @returns masks (Tensor): Tensor of masks of shape (batch_size, max_len)
    """
    masks = torch.ones(batch.size(0), batch.size(1), dtype=torch.float)
    for idx, src_len in enumerate(source_lengths):
        masks[idx, src_len:] = 0
    return masks.to(device)

def compute_mention_f1(y_true, y_pred):
    """Compute the mention F1 score as in CheXpert paper
    @param y_true (list): List of 14 tensors each of shape (dev_set_size)
    @param y_pred (list): Same as y_true but for model predictions

    @returns res (list): List of 14 scalars
    """
    for j in range(len(y_true)):
        y_true[j][y_true[j] == 2] = 1
        y_true[j][y_true[j] == 3] = 1
        y_pred[j][y_pred[j] == 2] = 1
        y_pred[j][y_pred[j] == 3] = 1

    res = []
    for j in range(len(y_true)): 
        res.append(f1_score(y_true[j], y_pred[j], pos_label=1))
        
    return res

def compute_blank_f1(y_true, y_pred):
    """Compute the blank F1 score 
    @param y_true (list): List of 14 tensors each of shape (dev_set_size)
    @param y_pred (list): Same as y_true but for model predictions
                                                                         
    @returns res (list): List of 14 scalars                           
    """
    for j in range(len(y_true)):
        y_true[j][y_true[j] == 2] = 1
        y_true[j][y_true[j] == 3] = 1
        y_pred[j][y_pred[j] == 2] = 1
        y_pred[j][y_pred[j] == 3] = 1

    res = []
    for j in range(len(y_true)):
        res.append(f1_score(y_true[j], y_pred[j], pos_label=0))

    return res
        
def compute_negation_f1(y_true, y_pred):
    """Compute the negation F1 score as in CheXpert paper
    @param y_true (list): List of 14 tensors each of shape (dev_set_size)
    @param y_pred (list): Same as y_true but for model predictions   

    @returns res (list): List of 14 scalars
    """
    for j in range(len(y_true)):
        y_true[j][y_true[j] == 3] = 0
        y_true[j][y_true[j] == 1] = 0
        y_pred[j][y_pred[j] == 3] = 0
        y_pred[j][y_pred[j] == 1] = 0

    res = []
    for j in range(len(y_true)-1):
        res.append(f1_score(y_true[j], y_pred[j], pos_label=2))

    res.append(0) #No Finding gets score of zero
    return res

def compute_positive_f1(y_true, y_pred):
    """Compute the positive F1 score
    @param y_true (list): List of 14 tensors each of shape (dev_set_size)
    @param y_pred (list): Same as y_true but for model predictions 

    @returns res (list): List of 14 scalars
    """
    for j in range(len(y_true)):
        y_true[j][y_true[j] == 3] = 0
        y_true[j][y_true[j] == 2] = 0
        y_pred[j][y_pred[j] == 3] = 0
        y_pred[j][y_pred[j] == 2] = 0

    res = []
    for j in range(len(y_true)):
        res.append(f1_score(y_true[j], y_pred[j], pos_label=1))

    return res
        
def compute_uncertain_f1(y_true, y_pred):
    """Compute the negation F1 score as in CheXpert paper
    @param y_true (list): List of 14 tensors each of shape (dev_set_size)
    @param y_pred (list): Same as y_true but for model predictions

    @returns res (list): List of 14 scalars
    """
    for j in range(len(y_true)):
        y_true[j][y_true[j] == 2] = 0
        y_true[j][y_true[j] == 1] = 0
        y_pred[j][y_pred[j] == 2] = 0
        y_pred[j][y_pred[j] == 1] = 0

    res = []
    for j in range(len(y_true)-1):
        res.append(f1_score(y_true[j], y_pred[j], pos_label=3))

    res.append(0) #No Finding gets a score of zero
    return res

def evaluate(model, dev_loader, device, f1_weights, return_pred=False):
    """ Function to evaluate the current model weights
    @param model (nn.Module): the labeler module 
    @param dev_loader (torch.utils.data.DataLoader): dataloader for dev set  
    @param device (torch.device): device on which data should be
    @param f1_weights (dictionary): dictionary mapping conditions to f1
                                    task weights
    @param return_pred (bool): whether to return predictions or not

    @returns res_dict (dictionary): dictionary with keys 'blank', 'mention', 'negation',
                           'uncertain', 'positive' and 'weighted', with values 
                            being lists of length 14 with each element in the 
                            lists as a scalar. If return_pred is true then a 
                            tuple is returned with the aforementioned dictionary 
                            as the first item, a list of predictions as the 
                            second item, and a list of ground truth as the 
                            third item
    """
    
    was_training = model.training
    model.eval()
    y_pred = [[] for _ in range(len(CONDITIONS))]
    y_true = [[] for _ in range(len(CONDITIONS))]
    
    with torch.no_grad():
        for i, data in enumerate(dev_loader, 0):
            batch = data['imp'] #(batch_size, max_len)
            batch = batch.to(device)
            label = data['label'] #(batch_size, 14)
            label = label.permute(1, 0).to(device)
            src_len = data['len']
            batch_size = batch.shape[0]
            attn_mask = generate_attention_masks(batch, src_len, device)

            out = model(batch, attn_mask)
            
            for j in range(len(out)):
                out[j] = out[j].to('cpu') #move to cpu for sklearn
                curr_y_pred = out[j].argmax(dim=1) #shape is (batch_size)
                y_pred[j].append(curr_y_pred)
                y_true[j].append(label[j].to('cpu'))

            if (i+1) % 200 == 0:
                print('Evaluation batch no: ', i+1)
                
    for j in range(len(y_true)):
        y_true[j] = torch.cat(y_true[j], dim=0)
        y_pred[j] = torch.cat(y_pred[j], dim=0)

    if was_training:
        model.train()

    mention_f1 = compute_mention_f1(copy.deepcopy(y_true), copy.deepcopy(y_pred))
    negation_f1 = compute_negation_f1(copy.deepcopy(y_true), copy.deepcopy(y_pred))
    uncertain_f1 = compute_uncertain_f1(copy.deepcopy(y_true), copy.deepcopy(y_pred))
    positive_f1 = compute_positive_f1(copy.deepcopy(y_true), copy.deepcopy(y_pred))
    blank_f1 = compute_blank_f1(copy.deepcopy(y_true), copy.deepcopy(y_pred))
    
    weighted = []
    kappas = []
    for j in range(len(y_pred)):
        cond = CONDITIONS[j]
        avg = weighted_avg([negation_f1[j], uncertain_f1[j], positive_f1[j]], f1_weights[cond])
        weighted.append(avg)

        mat = confusion_matrix(y_true[j], y_pred[j])
        kappas.append(cohens_kappa(mat, return_results=False))

    res_dict = {'mention': mention_f1,
                'blank': blank_f1,
                'negation': negation_f1,
                'uncertain': uncertain_f1,
                'positive': positive_f1,
                'weighted': weighted,
                'kappa': kappas}
    
    if return_pred:
        return res_dict, y_pred, y_true
    else:
        return res_dict

def test(model, checkpoint_path, test_ld, f1_weights):
    """Evaluate model on test set. 
    @param model (nn.Module): labeler module
    @param checkpoint_path (string): location of saved model checkpoint
    @param test_ld (dataloader): dataloader for test set
    @param f1_weights (dictionary): maps conditions to f1 task weights
    """
    # 修改设备为NPU
    device = torch.device("npu:0" if torch.npu.is_available() else "cpu")
    # 移除或注释掉 DataParallel
    # if torch.cuda.device_count() > 1: ...
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("Doing evaluation on test set\n")
    metrics = evaluate(model, test_ld, device, f1_weights)
    weighted = metrics['weighted']
    kappas = metrics['kappa']

    for j in range(len(CONDITIONS)):
        print('%s kappa: %.3f' % (CONDITIONS[j], kappas[j]))
    print('average: %.3f' % np.mean(kappas))

    print()
    for j in range(len(CONDITIONS)):
        print('%s weighted_f1: %.3f' % (CONDITIONS[j], weighted[j]))
    print('average of weighted_f1: %.3f' % (np.mean(weighted)))
    
    print()
    for j in range(len(CONDITIONS)):
        print('%s blank_f1:  %.3f, negation_f1: %.3f, uncertain_f1: %.3f, positive_f1: %.3f' % (CONDITIONS[j],
                                                                                                metrics['blank'][j],
                                                                                                metrics['negation'][j],
                                                                                                metrics['uncertain'][j],
                                                                                                metrics['positive'][j]))

    men_macro_avg = np.mean(metrics['mention'])
    neg_macro_avg = np.mean(metrics['negation'][:-1]) #No Finding has no negations
    unc_macro_avg = np.mean(metrics['uncertain'][:-2]) #No Finding, Support Devices have no uncertain labels in test set
    pos_macro_avg = np.mean(metrics['positive'])
    blank_macro_avg = np.mean(metrics['blank'])
        
    print("blank macro avg: %.3f, negation macro avg: %.3f, uncertain macro avg: %.3f, positive macro avg: %.3f" % (blank_macro_avg,
                                                                                                                    neg_macro_avg,
                                                                                                                    unc_macro_avg,
                                                                                                                    pos_macro_avg))
    print()
    for j in range(len(CONDITIONS)):
        print('%s mention_f1: %.3f' % (CONDITIONS[j], metrics['mention'][j]))
    print('mention macro avg: %.3f' % men_macro_avg)
    

def label_report_list(checkpoint_path, report_list):
    """ Evaluate model on list of reports.
    @param checkpoint_path (string): location of saved model checkpoint
    @param report_list (list): list of report impressions (string)
    """
    imp = pd.Series(report_list)
    imp = imp.str.strip()
    imp = imp.replace('\n',' ', regex=True)
    imp = imp.replace('[0-9]\.', '', regex=True)
    imp = imp.replace('\s+', ' ', regex=True)
    imp = imp.str.strip()
    
    model = bert_labeler()
    # 修改设备为NPU
    device = torch.device("npu:0" if torch.npu.is_available() else "cpu")
    # 移除或注释掉 DataParallel
    # if torch.cuda.device_count() > 1: ...
    model = model.to(device)
    # 加载checkpoint时，需要映射到当前设备
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 处理可能的 'module.' 前缀 (如果模型曾用DataParallel保存)
    new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(new_state_dict)

    model.eval()

    y_pred = []
    tokenizer = BertTokenizer.from_pretrained('/media/t1/zcy/dia-pangu/evaluate/bert-base-chinese-safetensors')
    new_imps = tokenize(imp, tokenizer)
    with torch.no_grad():
        for imp in new_imps:
            # run forward prop
            imp = torch.LongTensor(imp)
            source = imp.view(1, len(imp))
            
            attention = torch.ones(len(imp))
            attention = attention.view(1, len(imp))
            out = model(source.to(device), attention.to(device))

            # get predictions
            result = {}
            for j in range(len(out)):
                curr_y_pred = out[j].argmax(dim=1) #shape is (1)
                result[CONDITIONS[j]] = CLASS_MAPPING[curr_y_pred.item()]
            y_pred.append(result)
    return y_pred

