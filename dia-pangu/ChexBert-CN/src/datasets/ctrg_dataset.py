# datasets/ctrg_chexbert_dataset.py
import torch
import pandas as pd
import numpy as np
import json
from torch.utils.data import Dataset
from transformers import BertTokenizer

class CTRGCheXbertDataset(Dataset):
    """
    一个为CTRG-Chest数据集定制的Dataset类。
    它在运行时动态加载、匹配、转换数据，以适配CheXbert模型的训练需求。
    """
    def __init__(self, annotation_path, label_csv_path, split='train'):
        """
        初始化数据集。
        @param annotation_path (str): annotation.json 文件的路径。
        @param label_csv_path (str): CTRG_finding_labels.csv 文件的路径。
        @param split (str): 'train', 'val', 或 'test'，用于指定加载哪部分数据。
        """
        super().__init__()
        
        # 1. 加载并处理标签文件
        self.label_df = pd.read_csv(label_csv_path).set_index('id')
        self.conditions = list(self.label_df.columns) # 获取14个标签的列名

        # 2. 加载并处理报告原文文件
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotation_data = json.load(f)

        # 3. 根据split筛选样本ID，并构建ID到报告文本的映射
        self.sample_ids = []
        self.id_to_finding = {}
        
        # 遍历JSON中的'train', 'val', 'test'分区
        for part in annotation_data:
            for sample in annotation_data[part]:
                # 根据传入的split参数决定是否使用该样本
                # 注意：修复原始数据中'val'分区里split字段可能为'train'的问题
                current_sample_split = sample.get('split', part) # 以文件分区为准
                if current_sample_split == split:
                    sample_id_str = sample.get('id')
                    try:
                        # 确保CSV和JSON中的ID类型一致，这里都转为整数
                        sample_id_int = int(sample_id_str)
                        if sample_id_int in self.label_df.index:
                            self.sample_ids.append(sample_id_int)
                            self.id_to_finding[sample_id_int] = sample.get('finding', '')
                    except (ValueError, TypeError):
                        # 如果ID无法转换为整数，则跳过
                        continue
        
        # 4. 初始化中文分词器
        self.tokenizer = BertTokenizer.from_pretrained("/media/t1/zcy/dia-pangu/evaluate/bert-base-chinese-safetensors")

    def __len__(self):
        """返回数据集的样本总数"""
        return len(self.sample_ids)

    def __getitem__(self, idx):
        """
        获取单个数据样本。这是所有魔法发生的地方。
        @param idx (int): 样本索引。
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. 根据索引获取样本ID
        sample_id = self.sample_ids[idx]

        # 2. 获取报告原文并进行分词
        finding_text = self.id_to_finding.get(sample_id, '')
        # 清洗文本
        finding_text = finding_text.strip().replace('\n', ' ').replace('\s+', ' ')
        
        encoded_imp = self.tokenizer.encode_plus(
            finding_text,
            max_length=512,
            truncation=True,
            padding='max_length', # 直接在这里padding，简化collate_fn
            return_tensors='pt'
        )['input_ids'].squeeze(0) # 转换为 (512,) 的张量

        # 3. 获取原始标签 (值为 0, 1, 2)
        raw_labels = self.label_df.loc[sample_id]

        # 4. 在此动态转换标签以适配CheXbert模型的输入要求
        # CheXbert内部期望的标签: 0=Blank, 1=Positive, 2=Negative, 3=Uncertain
        # impressions_dataset.py 的转换逻辑:
        # CSV中的0 -> 2 (Negative)
        # CSV中的NaN -> 0 (Blank)
        # CSV中的1 -> 1 (Positive)
        # CSV中的-1 -> 3 (Uncertain)
        
        # 因此，我们需要准备一个 "输入给转换逻辑" 的Series
        mapped_labels = raw_labels.copy()
        
        # a. 转换前13个标签
        for cond in self.conditions[:-1]: # 排除 'no_finding'
            original_val = raw_labels[cond]
            if original_val == 1: # 存在 -> Positive
                mapped_labels[cond] = 1
            elif original_val == 2: # 不存在 -> Negative
                mapped_labels[cond] = 0 # 将在impressions_dataset的逻辑中被转为2
            elif original_val == 0: # 空白 -> Blank
                mapped_labels[cond] = np.nan # 将在impressions_dataset的逻辑中被转为0
        
        # b. 转换 'no_finding' 标签 (二分类)
        no_finding_val = raw_labels['no_finding']
        if no_finding_val == 1: # 存在
            mapped_labels['no_finding'] = 1
        else: # 不存在或空白
            mapped_labels['no_finding'] = 0

        # c. 应用 impressions_dataset.py 中的硬编码转换逻辑
        final_labels = mapped_labels.copy()
        final_labels.replace(0, 2, inplace=True)
        final_labels.fillna(0, inplace=True)
        
        # d. 修复 'no_finding' 列可能因转换产生的错误
        # 因为 no_finding 只需要 0 和 1，但上面的全局替换可能引入 2
        if final_labels['no_finding'] not in [0, 1]:
            final_labels['no_finding'] = 0 # 默认设为0
        
        final_labels_tensor = torch.LongTensor(final_labels.values.astype(int))

        return {
            "imp": encoded_imp,
            "label": final_labels_tensor,
            "len": (encoded_imp != self.tokenizer.pad_token_id).sum().item() # 计算真实长度
        }

