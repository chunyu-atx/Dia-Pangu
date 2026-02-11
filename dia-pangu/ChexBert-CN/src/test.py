# test.py
import argparse
import torch
import utils
from models.bert_labeler import bert_labeler
from datasets.ctrg_dataset import CTRGCheXbertDataset# 假设collate_fn也在那个文件里
from run_bert import collate_fn_labels
from constants import *


def load_test_data(annotation_path, label_csv_path, batch_size=BATCH_SIZE,
                   num_workers=NUM_WORKERS, shuffle=False):
    """为测试集创建 DataLoader"""
    collate_fn = collate_fn_labels
    # 使用 'test' split 来加载测试数据
    test_dset = CTRGCheXbertDataset(annotation_path, label_csv_path, split='test')
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=num_workers, collate_fn=collate_fn)
    return test_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate CheXbert on the test set.')
    parser.add_argument('--ann_path', type=str, required=True,
                        help='path to annotation.json file.')
    parser.add_argument('--label_path', type=str, required=True,
                        help='path to CTRG_finding_labels.csv file.')
    parser.add_argument('-c', '--checkpoint', type=str, required=True,
                        help='path to the model checkpoint to evaluate.')
    args = parser.parse_args()

    # 1. 加载测试数据
    print("Loading test data...")
    test_loader = load_test_data(args.ann_path, args.label_path)

    # 2. 初始化模型
    # 注意：这里我们不需要加载预训练权重，因为我们将从checkpoint加载所有权重
    model = bert_labeler(pretrain_path=PRETRAIN_PATH)

    # 3. 计算F1权重 (通常基于验证集或训练集，这里我们复用utils里的逻辑)
    # 注意：这里的f1_weights计算可能需要根据您的utils.py做相应调整
    # 假设f1_weights基于label_path的总体分布计算
    print("Calculating F1 weights...")
    f1_weights = utils.get_weighted_f1_weights(args.label_path)

    # 4. 调用 utils.py 中的 test 函数进行评估
    print("\nBegin evaluation on test set...")
    utils.test(model, args.checkpoint, test_loader, f1_weights)
