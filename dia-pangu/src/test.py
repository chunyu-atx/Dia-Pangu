import tqdm.auto as tqdm
import torch.nn.functional as F
from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
import transformers
from My_Trainer.trainer import Trainer
from dataclasses import dataclass, field
from Dataset.dataset_test import dataset_test
from Model.multimodality_model import MultiPanguForCausalLM
from datasampler import My_DistributedBatchSampler
import torch
from torch.utils.data import DataLoader
import csv
import random
import numpy as np
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
import os


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(20)
# 预处理数据以及训练模型


@dataclass
class ModelArguments:
    lang_encoder_path: Optional[str] = field(
        default="/media/t1/zcy/openPangu-Embedded-7B-V1.1")
    tokenizer_path: str = field(default="/media/t1/zcy/openPangu-Embedded-7B-V1.1",
                                metadata={"help": "Path to the tokenizer data."})


@dataclass
class DataArguments:
    Mode: Optional[str] = field(default="Train")
    test_split: Optional[str] = field(default="open")

def main():

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    device = torch.device(f"npu:{local_rank}")
    torch.npu.set_device(device)

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    print("Setup Data")
    Test_dataset = dataset_test(
        text_tokenizer=model_args.tokenizer_path
    )

    Test_dataloader = DataLoader(
        Test_dataset,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        sampler=None,
        shuffle=False,
        collate_fn=None,
        drop_last=False,
    )

    print("Setup Model")

    model = MultiPanguForCausalLM(
        text_tokenizer_path=model_args.tokenizer_path,
        lang_model_path=model_args.lang_encoder_path,
    )

    ckpt = torch.load(
        '/media/t1/zcy/dia-pangu/checkpoint_new/dia-pangu_v0112/pytorch_model.bin', map_location=device)
    model.load_state_dict(ckpt, strict=True)

    model = model.half()
    model.to(device)

    print("load ckpt")

    model.eval()

    with open('/media/t1/zcy/dia-pangu/results/' + 'dia-pangu'+'.csv', mode='w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(
            ["ID", "Question", "Ground Truth", "Pred", "GT_labels"])

        for sample in tqdm.tqdm(Test_dataloader):
            image_id = int(sample['id'][0])
            question = sample["question"]
            guide = sample['guide'][0]
            cls_labels = sample['cls_labels'][0]
            belong_to = sample['belong_to'][0]

            vision_x = sample["vision_x"].to(device).half()

            answer = sample['answer'][0]
            
            question, pred = model.generate(question, guide, vision_x)

            print('finding_gt: ', answer)
            print('finding_pred: ', pred)

            writer.writerow([image_id, question, answer, pred, cls_labels])



if __name__ == "__main__":
    main()
