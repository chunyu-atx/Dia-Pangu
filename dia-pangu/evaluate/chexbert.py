# chexbert.py (已修改)
from collections import OrderedDict
from transformers import BertConfig, BertModel, BertTokenizer
import os
import torch
import torch.nn as nn

# --- 关键：导入我们之前定义的常量 ---
# 这使得模型路径可以在一个地方集中管理
try:
    from constants import PRETRAIN_PATH
except ImportError:
    # 如果常量文件不存在，提供一个备用路径（请确保constants.py存在）
    PRETRAIN_PATH = "/media/t1/zcy/dia-pangu/evaluate/bert-base-chinese-safetensors"


class CheXbert(nn.Module):
    def __init__(self, checkpoint_path, device, p=0.1):
        super(CheXbert, self).__init__()

        self.device = device

        # --- 核心修改 (1): 加载您本地的中文模型和分词器 ---
        # 使用 PRETRAIN_PATH 变量，而不是硬编码的 'bert-base-uncased'
        print(f"[CheXbert Class] Initializing tokenizer from: {PRETRAIN_PATH}")
        self.tokenizer = BertTokenizer.from_pretrained(PRETRAIN_PATH)

        print(f"[CheXbert Class] Initializing BERT model from: {PRETRAIN_PATH}")
        # 直接从预训练路径加载已经带有权重的 BERT 模型，而不是只加载骨架
        self.bert = BertModel.from_pretrained(PRETRAIN_PATH)

        self.dropout = nn.Dropout(p)

        # 获取 BERT 模型的 hidden_size，这种方式更通用
        hidden_size = self.bert.config.hidden_size

        # 分类头的定义保持不变
        self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 4, bias=True) for _ in range(13)])
        self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))

        # --- 核心修改 (2): 简化并修正权重加载逻辑 ---
        # 加载您在中文数据上微调过的 checkpoint
        print(f"[CheXbert Class] Loading finetuned checkpoint: {checkpoint_path}")
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=device)['model_state_dict']

            # 我们自己训练的模型，key 已经是标准的，不需要复杂的替换逻辑
            # 只需要处理 'module.' 前缀（如果训练时用了 DataParallel）
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                # 只需移除 'module.' 前缀即可
                new_key = key.replace('module.', '')
                new_state_dict[new_key] = value

            # 使用 load_state_dict 加载权重
            # strict=False 允许只加载存在的key，忽略不匹配的key
            # 例如，BERT部分的权重已经从预训练加载，这里会用checkpoint里的权重覆盖它们
            self.load_state_dict(new_state_dict, strict=False)
            print("[CheXbert Class] Checkpoint loaded successfully.")
        else:
            print(f"WARNING: Checkpoint path not found: {checkpoint_path}. Using base BERT model only.")

        # 将整个模型移动到指定设备
        self.to(self.device)
        self.eval()

    def forward(self, reports):
        # 报告文本的清洗逻辑保持不变
        for i in range(len(reports)):
            reports[i] = reports[i].strip()
            reports[i] = reports[i].replace("\n", " ")
            # 修正正则表达式，\s+ 已经是匹配一个或多个空格
            reports[i] = " ".join(reports[i].split())

        with torch.no_grad():
            # 分词逻辑保持不变
            tokenized = self.tokenizer(reports, padding='longest', return_tensors="pt")
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

            # 模型前向传播逻辑保持不变
            last_hidden_state = self.bert(**tokenized)[0]

            cls = last_hidden_state[:, 0, :]
            cls = self.dropout(cls)

            predictions = []
            for i in range(14):
                predictions.append(self.linear_heads[i](cls).argmax(dim=1))

        return torch.stack(predictions, dim=1)

