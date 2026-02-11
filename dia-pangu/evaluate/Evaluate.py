# --- 确保在所有其他evaluate和nltk相关操作前，强制指定路径 ---
import nltk
import os
import sys

# 定义您的NLTK数据存放路径
custom_nltk_path = "/media/t1/zcy/dia-pangu/evaluate/nltk_data"

# 检查路径是否存在，如果存在，则将其强制插入到nltk的搜索路径列表的最前面
if os.path.exists(custom_nltk_path):
    # 使用 insert(0, ...) 来确保我们的路径是第一搜索顺位
    nltk.data.path.insert(0, custom_nltk_path)
    print(f"--- [NLTK Override] ---")
    print(f"Successfully forced NLTK data path to: {custom_nltk_path}")
    print(f"NLTK will now search in: {nltk.data.path}")
    print("-------------------------\n")
else:
    # 如果路径不存在，则打印错误并退出，防止后续卡死
    print(f"FATAL ERROR: Custom NLTK path not found: {custom_nltk_path}", file=sys.stderr)
    sys.exit(1)

from pprint import pprint
from metrics_clinical import CheXbertMetrics
import pandas as pd

# 导入评估所需库
import nltk
import evaluate
import warnings
import jieba
from bert_score import score
# 忽略一些不必要的警告
warnings.filterwarnings("ignore", category=UserWarning)

CONDITIONS = [
    '纵隔心影增大',
    '心脏肥大',
    '肺部阴影',
    '肺部病变',
    '水肿',
    '实变',
    '肺炎',
    '肺不张',
    '气胸',
    '胸腔积液',
    '胸膜其他病变',
    '骨折',
    '支撑装置',
    '未见异常',
]

os.environ["NLTK_DATA"] = "/media/t1/zcy/dia-pangu/evaluate/nltk_data"

def chinese_tokenizer(text):
    """为BLEU, METEOR, ROUGE提供jieba分词"""
    return " ".join(jieba.cut(text))


def main():
    # 加载结果到列表
    report_path = "/media/t1/zcy/dia-pangu/results/dia-pangu.csv"
    report_data = pd.read_csv(report_path)

    gt_list = report_data['Ground Truth'].tolist()
    res_list = report_data['Pred'].tolist()


    # ==================== 评估部分 (已更新) ====================
    print("\n" + "=" * 40)
    print("      正在计算评估指标...      ")
    print("=" * 40)

    # --- 为BLEU和ROUGE准备经过jieba分词、用空格连接的字符串 ---
    predictions_for_bleu = [chinese_tokenizer(p) for p in res_list]
    references_for_bleu = [[chinese_tokenizer(r)] for r in gt_list]

    # --- BLEU 指标 ---
    print("--> Calculating BLEU...")
    bleu_metric = evaluate.load('/media/t1/zcy/dia-pangu/evaluate/bleu/bleu.py')  # 使用本地路径
    bleu_scores = bleu_metric.compute(
        predictions=predictions_for_bleu,
        references=references_for_bleu
    )
    print("    BLEU calculated successfully.")

    # --- METEOR 指标 ---
    print("--> Calculating METEOR")
    meteor_metric = evaluate.load('/media/t1/zcy/dia-pangu/evaluate/meteor/meteor.py')  # 使用本地路径
    meteor_scores = meteor_metric.compute(
        predictions=res_list,  # METEOR可以直接处理字符串
        references=gt_list
    )
    print("    METEOR calculated successfully.")

    # --- ROUGE 指标 ---
    print("--> Calculating ROUGE...")
    rouge_metric = evaluate.load('/media/t1/zcy/dia-pangu/evaluate/rouge/rouge.py')  # 使用本地路径
    # ROUGE可以直接传入tokenizer函数
    rouge_scores = rouge_metric.compute(
        predictions=res_list,
        references=gt_list,
        tokenizer=chinese_tokenizer
    )
    print("    ROUGE calculated successfully.")

    # --- BERTScore 指标 (作为补充) ---
    (P, R, F1) = score(
        cands=res_list,
        refs=gt_list,
        lang="zh",
        model_type='/media/t1/zcy/dia-pangu/evaluate/bert-base-chinese-safetensors',
        num_layers=12,
        rescale_with_baseline=False,
        verbose=True,
        device='npu'
    )
    avg_bert_f1 = F1.mean().item()


    # ==================== CheXbert 临床准确性评估 (核心修改部分) ====================
    print("\n" + "=" * 40)
    print("  正在计算 CheXbert 临床准确性指标...  ")
    print("=" * 40)

    # 指定您训练好的、最好的中文 CheXbert 模型 checkpoint 路径
    # 假设 model_epoch8_iter200 是最好的
    finetuned_chexbert_path = "/media/t1/zcy/CheXbert-master/ckpt/model_epoch8_iter200"

    # 初始化 CheXbertMetrics 评估器
    # 这里的 mbatch_size 可以根据您的NPU显存调整
    chexbert_metrics = CheXbertMetrics(
        checkpoint_path=finetuned_chexbert_path,
        mbatch_size=16,
        device='npu'
    )

    # 执行计算，得到分数
    # ce_scores 是包含宏/微平均的字典
    # f1_class 是一个包含14个标签各自F1分数的数组
    ce_scores, f1_class = chexbert_metrics.compute(gt_list, res_list)

    # --- 统一打印所有NLG指标 ---
    print("\n--- NLG 评估结果 ---")
    print(f"BLEU-1: {bleu_scores['precisions'][0]:.4f}")
    print(f"BLEU-2: {bleu_scores['precisions'][1]:.4f}")
    print(f"BLEU-3: {bleu_scores['precisions'][2]:.4f}")
    print(f"BLEU-4: {bleu_scores['precisions'][3]:.4f}")
    print(f"METEOR: {meteor_scores['meteor']:.4f}")
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
    print(f"BERTScore F1 (补充): {avg_bert_f1:.4f}")
    print("\n" + "=" * 40)

    # --- 打印结果 (增强版) ---
    print("\n--- 临床实体 (CE) 评估结果 ---")

    # --- 核心单一评估指标 ---
    print("\n[核心单一评估指标]")
    print("---------------------------------")
    print(f"宏平均 F1-Score (Macro-F1):   {ce_scores['ce_f1_macro']:.4f}")
    print(f"微平均 F1-Score (Micro-F1):   {ce_scores['ce_f1_micro']:.4f}")
    print("---------------------------------")
    print("(注：Micro-F1更能代表模型在整个数据集上的总体表现)")

    # --- 各类别详细F1分数 ---
    print("\n[各类别 F1-Score (用于诊断模型强弱项)]")
    # 创建一个 DataFrame 来美观地展示每个类别的F1分数
    f1_df = pd.DataFrame({'Condition': CONDITIONS, 'F1-Score': f1_class})
    print(f1_df.to_string(index=False))

    # --- 其他详细指标 (可选) ---
    print("\n[其他详细整体指标]")
    # 使用 pprint 获得更美观的字典输出
    pprint(ce_scores)

    print("\n" + "=" * 40)




if __name__ == '__main__':
    main()
