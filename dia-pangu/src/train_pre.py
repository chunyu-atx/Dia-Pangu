import tqdm.auto as tqdm
import torch.nn.functional as F
from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
import transformers

import torch.distributed as dist

# 如果使用 torchrun，则需要手动指定 HCCL backend
if dist.is_available() and not dist.is_initialized():
    dist.init_process_group(backend="hccl")

from My_Trainer.trainer import Trainer
from dataclasses import dataclass, field
from Model.multimodality_model import MultiPanguForCausalLM
from datasampler import My_DistributedBatchSampler
from Dataset.dataset_train import dataset_train
from Dataset.dataset_val import dataset_val
import numpy as np
import torch
import random
import os

try:
    import torch_npu
    npu_available = True
except ImportError:
    npu_available = False

def log_device(training_args):
    """Prints information about the hardware and distributed environment."""
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        distributed = True
        dist_url = os.environ.get("MASTER_ADDR", "N/A")
    else:
        rank = 0
        world_size = 1
        distributed = False
        dist_url = "N/A"
    print(f"[dist] rank={rank}, world={world_size}, distributed={distributed}, dist_url={dist_url}")
    print(f"[env] RANK={os.environ.get('RANK')} LOCAL_RANK={os.environ.get('LOCAL_RANK')} WORLD_SIZE={os.environ.get('WORLD_SIZE')}")
    print(f"[torch] torch={torch.__version__} torch_npu={getattr(torch_npu, '__version__', 'not installed')}")
    if training_args.device.type == "npu" and npu_available:
        idx = torch.npu.current_device()
        props = torch.npu.get_device_properties(idx)
        print(f"[device] NPU index={idx} name={torch.npu.get_device_name(idx)} total_mem={props.total_memory/1024**3:.1f}GB count={torch.npu.device_count()}")
    elif torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        print(f"[device] CUDA index={idx} name={torch.cuda.get_device_name(idx)} total_mem={props.total_memory/1024**3:.1f}GB count={torch.cuda.device_count()}")
    else:
        print("[device] CPU mode")

def log_model(model):
    """Prints information about the language model and tokenizer."""
    print(f"[model] top_class={model.__class__.__name__}")
    # Check for lang_model, llm, or model attributes
    llm = getattr(model, "lang_model", None) or getattr(model, "llm", None) or getattr(model, "model", None)
    if llm is not None:
        try:
            p = next(iter(llm.parameters()))
            total = sum(p.numel() for p in llm.parameters()) / 1e9
            trainable = sum(p.numel() for p in llm.parameters() if p.requires_grad) / 1e9
            print(f"[model] llm_class={llm.__class__.__name__} dtype={p.dtype} device={p.device} params={total:.2f}B trainable={trainable:.2f}B")

            if hasattr(llm, 'config'):
                print(f"[model] llm_config_class={llm.config.__class__.__name__}")
                print(llm.config.to_json_string())
            else:
                print("[model] llm_config not found.")
        except StopIteration:
            print("[model] LLM has no parameters.")
    # Check for text_tokenizer or tokenizer attributes
    tok = getattr(model, "text_tokenizer", None) or getattr(model, "tokenizer", None)
    if tok is not None:
        print(f"[tokenizer] vocab={tok.vocab_size} pad={tok.pad_token_id} eos={tok.eos_token_id} file={getattr(tok, 'vocab_file', 'n/a')}")

def compute_metrics(eval_preds):
    # metric = load_metric("glue", "mrpc")
    ACCs = eval_preds.predictions
    # print(ACCs)
    return {"accuracy": np.mean(ACCs, axis=-1)}


@dataclass
class ModelArguments:
    lang_encoder_path: Optional[str] = field(
        default="/media/t1/zcy/openPangu-Embedded-7B-V1.1")
    tokenizer_path: str = field(default="/media/t1/zcy/openPangu-Embedded-7B-V1.1",
                                metadata={"help": "Path to the tokenizer data."})


@dataclass
class DataArguments:
    Mode: Optional[str] = field(default="Train")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    batch_size_2D: int = field(default=4)
    batch_size_3D: int = field(default=1)
    output_dir: Optional[str] = field(
        default="/media/t1/zcy/dia-pangu/checkpoint_new/")
    cache_dir: Optional[str] = field(default="/media/t1/zcy/openPangu-Embedded-7B-V1.1")
    optim: str = field(default="adamw_torch")
    bf16: bool = field(default=True)  # 启用bf16混合精度
    fp16: bool = field(default=False)

    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=8)


@dataclass
class DataCollator(object):

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # print(instances) 'loss_reweight': reweight_tensor, 'key_words_query': emphasize_words
        vision_xs, lang_xs, cls_labels, attention_masks, labels, loss_reweight, key_words_query = tuple([instance[key] for instance in instances] for key in (
            'vision_x', 'lang_x', 'cls_labels', 'attention_mask', 'labels', 'loss_reweight', 'key_words_query'))

        lang_xs = torch.cat([_.unsqueeze(0) for _ in lang_xs], dim=0)
        cls_labels = torch.cat([_.unsqueeze(0) for _ in cls_labels], dim=0)
        attention_masks = torch.cat([_.unsqueeze(0)
                                    for _ in attention_masks], dim=0)
        labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
        loss_reweight = torch.cat([_.unsqueeze(0)
                                  for _ in loss_reweight], dim=0)
        # print(lang_xs.shape,attention_masks.shape,labels.shape)

        target_H = 512
        target_W = 512
        target_D = 4
        MAX_D = 0

        D_list = list(range(4, 65, 4))
        if len(vision_xs) == 1:
            if vision_xs[0].shape[0] > 6:
                D_list = list(range(4, 33, 4))

        for ii in vision_xs:
            try:
                D = ii.shape[-1]
                if D > MAX_D:
                    MAX_D = D
            except:
                continue
        for temp_D in D_list:
            if abs(temp_D - MAX_D) < abs(target_D - MAX_D):
                target_D = temp_D

        if len(vision_xs) == 1 and target_D > 4:
            target_H = 256
            target_W = 256

        vision_xs = [torch.nn.functional.interpolate(
            s, size=(target_H, target_W, target_D)) for s in vision_xs]

        vision_xs = torch.nn.utils.rnn.pad_sequence(
            vision_xs, batch_first=True, padding_value=0
        )
        # print(vision_xs.shape,vision_xs.dtype)
        return dict(
            lang_x=lang_xs,
            vision_x=vision_xs,
            cls_labels=cls_labels,
            attention_mask=attention_masks,
            labels=labels,
            loss_reweight=loss_reweight,
            key_words_query=key_words_query
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(42)
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # --- Print device and model info ---
    log_device(training_args)
    # -----------------------------------

    training_args.data_sampler = My_DistributedBatchSampler

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print("Setup Data")
    Train_dataset = dataset_train(
        text_tokenizer=model_args.tokenizer_path,
    )

    Eval_dataset = dataset_val(
        text_tokenizer=model_args.tokenizer_path,
    )

    print("Setup Model")

    model = MultiPanguForCausalLM(
        lang_model_path=model_args.lang_encoder_path,
        text_tokenizer_path=model_args.tokenizer_path,
    )

    # --- Print device and model info ---
    log_model(model)
    # -----------------------------------

    # load model weights
    ckpt = torch.load('/media/t1/zcy/dia-pangu/checkpoint/pytorch_model.bin', map_location='npu')
    # model.load_state_dict(ckpt, strict=False)

    current_model_dict = model.state_dict()


    new_state_dict = {k: v for k, v in ckpt.items() if
                      k in current_model_dict and v.shape == current_model_dict[k].shape}

    print("Weights discarded due to shape mismatch:")
    for k, v in ckpt.items():
        if k in current_model_dict and v.shape != current_model_dict[k].shape:
            print(f"  - {k}: Checkpoint shape {v.shape}, Model shape {current_model_dict[k].shape}")

    model.load_state_dict(new_state_dict, strict=False)

    print("\nSuccessfully loaded compatible weights!")

    print("Model weights loaded successfully")


    ###############################################################
    # Freeze LLM and other modules — only train Perceiver
    ###############################################################
    print("Freezing all parameters except Perceiver...")

    trainable_param_names = []

    for name, param in model.named_parameters():
        # 只训练 Perceiver
        if name.startswith("perceiver") or "perceiver" in name:
            param.requires_grad = True
            trainable_param_names.append(name)
        else:
            param.requires_grad = False

    print("Trainable parameters:")
    for n in trainable_param_names:
        print("  ", n)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total/1e6:.2f}M  — Trainable: {trainable/1e6:.2f}M")


    trainer = Trainer(model=model,
                      train_dataset=Train_dataset,
                      eval_dataset=Eval_dataset,
                      args=training_args,
                      data_collator=DataCollator(),
                      local_rank=local_rank,
                      )

    trainer.train(epochs=int(training_args.num_train_epochs))
    trainer.save(training_args.output_dir)


if __name__ == "__main__":
    main()
