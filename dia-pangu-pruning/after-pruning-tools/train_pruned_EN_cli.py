# train_pruned_v0112_ENcli.py
import os
import json
import math
import time
import random
import re
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import transformers


# ========= Patch accelerate (avoid peft ImportError) =========
def patch_accelerate_clear_device_cache():
    """
    peft's loftq_utils may import: from accelerate.utils.memory import clear_device_cache
    Some accelerate versions don't have it -> ImportError.
    We inject a compatible function before peft is imported anywhere.
    """
    try:
        import accelerate.utils.memory as am
        if not hasattr(am, "clear_device_cache"):
            def clear_device_cache():
                try:
                    if hasattr(torch, "cuda") and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    if hasattr(torch, "npu"):
                        torch.npu.empty_cache()
                except Exception:
                    pass
            am.clear_device_cache = clear_device_cache
    except Exception:
        pass

patch_accelerate_clear_device_cache()

from My_Trainer.trainer import Trainer
from datasampler import My_DistributedBatchSampler
from Dataset_EN.dataset_train import dataset_train
from Dataset_EN.dataset_val import dataset_val

from Model.multimodality_model_pruned_v0 import (
    MultiPanguForCausalLMPruned,
    smart_load_state_dict,
    get_rank0,
)

# ======== v0112 ckpt keys ========
LLM_LAYER_PREFIX = "lang_model.base_model.model.model.layers."
LLM_LAYER_RE = re.compile(r"^" + re.escape(LLM_LAYER_PREFIX) + r"(\d+)\.")

def maybe_init_dist():
    """
    兼容两种启动方式：
    - torchrun 单卡/多卡：RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT 都在 -> init
    - 直接 python 跑：没有这些 env -> 不 init（避免卡死）
    """
    if not dist.is_available() or dist.is_initialized():
        return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "hccl" if torch.npu.is_available() else ("nccl" if torch.cuda.is_available() else "gloo")
        dist.init_process_group(backend=backend)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.npu.is_available():
        torch.npu.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_layers_spec(spec: str) -> List[int]:
    """
    支持：
      - "23-28"
      - "23,24,25"
      - "0-2,5,7-9"
    """
    spec = (spec or "").strip()
    if not spec:
        return []
    parts = re.split(r"[,\s]+", spec)
    out: List[int] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-", 1)
            a, b = a.strip(), b.strip()
            if a == "" or b == "":
                raise ValueError(f"Bad range token: {p}")
            start, end = int(a), int(b)
            if start > end:
                raise ValueError(f"Range must be ascending: {p}")
            out.extend(list(range(start, end + 1)))
        else:
            out.append(int(p))
    return sorted(set(out))

def make_drop_tag(drop_layers: List[int]) -> str:
    if not drop_layers:
        return "dropNone"
    s = sorted(drop_layers)
    is_contig = all(s[i] + 1 == s[i + 1] for i in range(len(s) - 1))
    if is_contig:
        return f"drop{s[0]}_{s[-1]}"
    return "drop" + "_".join(str(x) for x in s)

def synchronize_device():
    if torch.npu.is_available():
        try:
            torch.npu.synchronize()
        except Exception:
            pass
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

def try_get_train_dataloader_len(trainer) -> Optional[int]:
    candidates = []
    if hasattr(trainer, "get_train_dataloader"):
        candidates.append(("get_train_dataloader", trainer.get_train_dataloader))
    if hasattr(trainer, "train_dataloader"):
        candidates.append(("train_dataloader", lambda: trainer.train_dataloader))

    for _, fn in candidates:
        try:
            dl = fn()
            if dl is None:
                continue
            return len(dl)
        except Exception:
            continue
    return None

def get_global_step(trainer) -> Optional[int]:
    st = getattr(trainer, "state", None)
    if st is not None and hasattr(st, "global_step"):
        try:
            return int(st.global_step)
        except Exception:
            pass
    if hasattr(trainer, "global_step"):
        try:
            return int(getattr(trainer, "global_step"))
        except Exception:
            pass
    return None

def prepare_output_dir(output_dir: str) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    meta_dir = os.path.join(output_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    return {
        "output_dir": output_dir,
        "meta_dir": meta_dir,
        "run_started_json": os.path.join(meta_dir, "run_started.json"),
        "lora_meta_json": os.path.join(meta_dir, "lora_meta.json"),
        "timing_json": os.path.join(output_dir, "timing.json"),
        "interrupt_flag": os.path.join(meta_dir, "INTERRUPTED"),
    }

def safe_write_json(path: str, obj: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def safe_trainer_save(trainer, output_dir: str):
    if get_rank0() == 0:
        print(f"[SAVE] begin saving to: {output_dir}", flush=True)
    t0 = time.perf_counter()
    trainer.save(output_dir)
    t1 = time.perf_counter()
    if get_rank0() == 0:
        print(f"[SAVE] done. elapsed={t1 - t0:.2f}s", flush=True)

def load_adapter_config_if_exists(ckpt_path: str) -> Optional[dict]:
    ckpt_dir = os.path.dirname(ckpt_path)
    cand = os.path.join(ckpt_dir, "adapter_config.json")
    if os.path.isfile(cand):
        try:
            with open(cand, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def infer_lora_meta_from_ckpt_v0112(sd: Dict[str, torch.Tensor]) -> Tuple[List[str], int]:
    """
    v0112：从 keys 推断 LoRA 的 target_modules 和 rank r
      lang_model.base_model.model.model.layers.<i>.self_attn.q_proj.lora_A.default.weight -> target=q_proj, r=shape[0]
    """
    targets = set()
    r_set = set()
    pat = re.compile(
        r"^" + re.escape(LLM_LAYER_PREFIX) + r"\d+\.self_attn\.([a-zA-Z0-9_]+)\.lora_A\.default\.weight$"
    )
    for k, v in sd.items():
        m = pat.match(k)
        if m and hasattr(v, "shape"):
            targets.add(m.group(1))
            r_set.add(int(v.shape[0]))

    if not targets:
        raise RuntimeError("Cannot infer LoRA target_modules: no self_attn.*.lora_A.default.weight found in ckpt.")
    if len(r_set) != 1:
        raise RuntimeError(f"LoRA rank inconsistent across modules: {sorted(list(r_set))}")
    return sorted(list(targets)), int(next(iter(r_set)))

def normalize_state_dict_obj(ckpt_obj):
    """
    支持：
      - ckpt_obj 直接是 state_dict
      - ckpt_obj['state_dict'] / ['model'] / ['module']
    并顺手去掉 'module.' 前缀
    """
    if isinstance(ckpt_obj, dict) and any(hasattr(v, "shape") for v in ckpt_obj.values()):
        sd = ckpt_obj
    elif isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model", "module"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                sd = ckpt_obj[k]
                break
        else:
            sd = ckpt_obj
    else:
        sd = ckpt_obj

    if isinstance(sd, dict):
        sd = {kk[7:] if kk.startswith("module.") else kk: vv for kk, vv in sd.items()}
    return sd

def load_prune_map(prune_map_path: str) -> dict:
    with open(prune_map_path, "r", encoding="utf-8") as f:
        return json.load(f)

@dataclass
class ModelArguments:
    lang_encoder_path: Optional[str] = field(default="/media/t1/zcy/openPangu-Embedded-7B-V1.1")
    tokenizer_path: str = field(default="/media/t1/zcy/openPangu-Embedded-7B-V1.1")

    # 剪枝后合并 clean 的权重
    pruned_model_path: str = field(default="")

    # 你可以直接写 "23-28" 或 "23,24,..."（若提供 prune_map_path 则可不填）
    drop_layers: str = field(default="")

    # 如果你给 prune_map_path，就从里面读 drop_layers（优先）
    prune_map_path: str = field(default="")

    # ===== LoRA finetuning args (manual, from CLI) =====
    lora_target_modules: str = field(default="q_proj,v_proj")  # comma-separated
    lora_r: int = field(default=4)
    lora_alpha: int = field(default=8)
    lora_dropout: float = field(default=0.2)

@dataclass
class DataArguments:
    Mode: Optional[str] = field(default="Train")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    batch_size_2D: int = field(default=4)
    batch_size_3D: int = field(default=1)

    # 如果你不传 output_dir，可以传 out_root，让脚本自动生成 output_dir=<out_root>/ft_out_v0112_<tag>
    output_dir: Optional[str] = field(default="")
    out_root: Optional[str] = field(default="/media/t1/sym/workspace_prune_new")

    optim: str = field(default="adamw_torch")
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)

    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=8)

    save_strategy: str = field(default="steps")
    save_steps: int = field(default=200)
    save_total_limit: int = field(default=2)

@dataclass
class DataCollator(object):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        vision_xs, lang_xs, cls_labels, attention_masks, labels, loss_reweight, key_words_query = tuple(
            [instance[key] for instance in instances]
            for key in (
                "vision_x",
                "lang_x",
                "cls_labels",
                "attention_mask",
                "labels",
                "loss_reweight",
                "key_words_query",
            )
        )

        lang_xs = torch.cat([_.unsqueeze(0) for _ in lang_xs], dim=0)
        cls_labels = torch.cat([_.unsqueeze(0) for _ in cls_labels], dim=0)
        attention_masks = torch.cat([_.unsqueeze(0) for _ in attention_masks], dim=0)
        labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
        loss_reweight = torch.cat([_.unsqueeze(0) for _ in loss_reweight], dim=0)

        target_H, target_W, target_D = 512, 512, 4
        MAX_D = 0
        D_list = list(range(4, 65, 4))
        if len(vision_xs) == 1 and vision_xs[0].shape[0] > 6:
            D_list = list(range(4, 33, 4))

        for ii in vision_xs:
            try:
                D = ii.shape[-1]
                MAX_D = max(MAX_D, D)
            except Exception:
                continue
        for temp_D in D_list:
            if abs(temp_D - MAX_D) < abs(target_D - MAX_D):
                target_D = temp_D

        if len(vision_xs) == 1 and target_D > 4:
            target_H, target_W = 256, 256

        vision_xs = [torch.nn.functional.interpolate(s, size=(target_H, target_W, target_D)) for s in vision_xs]
        vision_xs = torch.nn.utils.rnn.pad_sequence(vision_xs, batch_first=True, padding_value=0)

        return dict(
            lang_x=lang_xs,
            vision_x=vision_xs,
            cls_labels=cls_labels,
            attention_mask=attention_masks,
            labels=labels,
            loss_reweight=loss_reweight,
            key_words_query=key_words_query,
        )

def main():
    maybe_init_dist()
    set_seed(42)

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # ===== Required base model/tokenizer dir (EN pipeline) =====
    REQUIRED_BASE_MODEL_DIR = "/media/t1/zcy/openPangu-Embedded-7B-V1.1"
    if (model_args.lang_encoder_path or "").strip() != REQUIRED_BASE_MODEL_DIR or (model_args.tokenizer_path or "").strip() != REQUIRED_BASE_MODEL_DIR:
        raise ValueError(
            "For EN reusable scripts, lang_encoder_path and tokenizer_path must be fixed to: "
            f"{REQUIRED_BASE_MODEL_DIR}"
        )

    # ===== Required inputs from CLI =====
    if not (model_args.pruned_model_path or "").strip():
        raise ValueError("--pruned_model_path is required (path to pruned merged clean weights)")
    if not (model_args.prune_map_path or "").strip():
        raise ValueError("--prune_map_path is required (path to prune_layer_map json)")


    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.npu.is_available():
        torch.npu.set_device(local_rank)
        device = torch.device("npu", local_rank)
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    # ===== 确定 drop_layers（prune_map_path 优先）=====
    if model_args.prune_map_path.strip():
        pm = load_prune_map(model_args.prune_map_path)
        drop_layers_list = pm.get("drop_layers", [])
        if not isinstance(drop_layers_list, list):
            raise ValueError("prune_map_path.drop_layers must be a list")
        drop_layers = sorted(set(int(x) for x in drop_layers_list))
        tag = pm.get("tag", make_drop_tag(drop_layers))
    else:
        drop_layers = parse_layers_spec(model_args.drop_layers)
        tag = make_drop_tag(drop_layers)

    if not model_args.pruned_model_path.strip():
        raise ValueError("--pruned_model_path is required")

    # ===== 自动 output_dir（如果没传）=====
    if not training_args.output_dir or str(training_args.output_dir).strip() == "":
        out_root = training_args.out_root or "."
        training_args.output_dir = os.path.join(out_root, f"ft_out_v0112_{tag}")

    # 先创建目录，写 meta
    paths = prepare_output_dir(training_args.output_dir)

    if get_rank0() == 0:
        run_started = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "device": str(device),
            "local_rank": local_rank,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "output_dir": training_args.output_dir,
            "lang_encoder_path": model_args.lang_encoder_path,
            "tokenizer_path": model_args.tokenizer_path,
            "pruned_model_path": model_args.pruned_model_path,
            "drop_layers": drop_layers,
            "tag": tag,
            "training_args": {
                "num_train_epochs": float(training_args.num_train_epochs),
                "per_device_train_batch_size": int(training_args.per_device_train_batch_size),
                "gradient_accumulation_steps": int(training_args.gradient_accumulation_steps),
                "bf16": bool(training_args.bf16),
                "fp16": bool(training_args.fp16),
                "save_strategy": str(getattr(training_args, "save_strategy", "")),
                "save_steps": int(getattr(training_args, "save_steps", -1)),
            },
        }
        safe_write_json(paths["run_started_json"], run_started)
        print(f"[env] rank={get_rank0()} local_rank={local_rank} device={device}", flush=True)
        print(f"[ckpt] pruned_model_path={model_args.pruned_model_path}", flush=True)
        print(f"[prune] drop_layers={drop_layers} tag={tag}", flush=True)
        print(f"[args] output_dir={training_args.output_dir}", flush=True)

    training_args.data_sampler = My_DistributedBatchSampler

    if get_rank0() == 0:
        print("[data] building datasets...", flush=True)
    Train_dataset = dataset_train(text_tokenizer=model_args.tokenizer_path)
    Eval_dataset = dataset_val(text_tokenizer=model_args.tokenizer_path)

    # ===== 先加载 ckpt（CPU）用来推断 LoRA meta =====
    if get_rank0() == 0:
        print("[ckpt] loading checkpoint to CPU for meta-infer...", flush=True)
    ckpt_obj = torch.load(model_args.pruned_model_path, map_location="cpu")
    ckpt_sd = normalize_state_dict_obj(ckpt_obj)

    inferred_targets, inferred_r = infer_lora_meta_from_ckpt_v0112(ckpt_sd)
    adapter_cfg = load_adapter_config_if_exists(model_args.pruned_model_path)

    # ===== Use LoRA args from CLI (manual) =====
    target_modules = [x.strip() for x in (model_args.lora_target_modules or "").split(",") if x.strip()]
    lora_r = int(model_args.lora_r)
    lora_alpha = int(model_args.lora_alpha)
    lora_dropout = float(model_args.lora_dropout)

    # Optional: compare with ckpt/adapter_config for early warning (rank0 only)
    if get_rank0() == 0:
        ckpt_targets = inferred_targets
        ckpt_r = inferred_r
        if adapter_cfg is not None:
            try:
                ckpt_targets = adapter_cfg.get("target_modules", ckpt_targets)
                if isinstance(ckpt_targets, (set, tuple)):
                    ckpt_targets = list(ckpt_targets)
                ckpt_targets = list(ckpt_targets)
                ckpt_r = int(adapter_cfg.get("r", ckpt_r))
            except Exception:
                pass

        if ckpt_r != lora_r:
            print(
                f"[WARN][lora] CLI lora_r={lora_r} differs from ckpt inferred r={ckpt_r}. "
                f"If load_state_dict fails due to shape mismatch, set --lora_r {ckpt_r}.",
                flush=True,
            )
        if sorted(ckpt_targets) != sorted(target_modules):
            print(
                f"[WARN][lora] CLI target_modules={target_modules} differs from ckpt inferred targets={ckpt_targets}. "
                f"If load_state_dict fails, set --lora_target_modules \"{','.join(ckpt_targets)}\".",
                flush=True,
            )

        safe_write_json(
            paths["lora_meta_json"],
            {
                "target_modules": target_modules,
                "r": int(lora_r),
                "alpha": int(lora_alpha),
                "dropout": float(lora_dropout),
                "note": "LoRA config is taken from CLI args; ckpt/adapter_config only used for warnings.",
            },
        )
        print(
            f"[lora-meta] target_modules={target_modules}, r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout} (from CLI)",
            flush=True,
        )

    # ===== 构建模型（按 drop_layers 重建 pruned config），并 wrap LoRA =====
    if get_rank0() == 0:
        print("[model] building pruned model (v0112)...", flush=True)

    model = MultiPanguForCausalLMPruned(
        lang_model_path=model_args.lang_encoder_path,
        text_tokenizer_path=model_args.tokenizer_path,
        drop_layers=drop_layers,
        device=device,
        use_bf16=bool(training_args.bf16),
        lora_target_modules=target_modules,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    # ===== 加载剪枝后合并权重 =====
    if get_rank0() == 0:
        print("[ckpt] loading pruned merged checkpoint weights...", flush=True)
    smart_load_state_dict(model, ckpt_obj, drop_layers=drop_layers)
    if get_rank0() == 0:
        print("[ckpt] done.", flush=True)

    trainer = Trainer(
        model=model,
        train_dataset=Train_dataset,
        eval_dataset=Eval_dataset,
        args=training_args,
        data_collator=DataCollator(),
        local_rank=local_rank,
    )

    micro_steps_per_epoch = try_get_train_dataloader_len(trainer)
    if micro_steps_per_epoch is None:
        micro_steps_per_epoch = math.ceil(len(Train_dataset) / max(1, int(training_args.per_device_train_batch_size)))
        if get_rank0() == 0:
            print("[step] WARNING: cannot get len(train_dataloader) from Trainer; using rough estimate.", flush=True)

    opt_steps_per_epoch = math.ceil(micro_steps_per_epoch / max(1, int(training_args.gradient_accumulation_steps)))
    total_epochs = int(training_args.num_train_epochs)
    planned_opt_steps = opt_steps_per_epoch * total_epochs

    if get_rank0() == 0:
        print(f"[step] micro_steps_per_epoch={micro_steps_per_epoch}", flush=True)
        print(f"[step] optimizer_steps_per_epoch≈ceil({micro_steps_per_epoch}/{training_args.gradient_accumulation_steps})={opt_steps_per_epoch}", flush=True)
        print(f"[step] planned_optimizer_steps≈{planned_opt_steps} (epochs={total_epochs})", flush=True)

    interrupted = False
    failed_exc = None

    synchronize_device()
    t0 = time.perf_counter()

    try:
        trainer.train(epochs=total_epochs)
    except KeyboardInterrupt:
        interrupted = True
        if get_rank0() == 0:
            with open(paths["interrupt_flag"], "w", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            print("\n[INTERRUPT] Caught Ctrl+C. Will save checkpoint before exit.", flush=True)
    except Exception as e:
        failed_exc = repr(e)
        if get_rank0() == 0:
            print(f"\n[ERROR] Training failed with exception: {failed_exc}", flush=True)
        raise
    finally:
        synchronize_device()
        t1 = time.perf_counter()
        train_seconds = t1 - t0

        gs = get_global_step(trainer)

        if get_rank0() == 0:
            try:
                safe_trainer_save(trainer, training_args.output_dir)
            except Exception as e:
                print(f"[SAVE][ERROR] failed to save checkpoint: {repr(e)}", flush=True)

            timing = {
                "status": "interrupted" if interrupted else ("failed" if failed_exc else "finished"),
                "exception": failed_exc,
                "num_train_epochs": total_epochs,
                "micro_steps_per_epoch": int(micro_steps_per_epoch),
                "gradient_accumulation_steps": int(training_args.gradient_accumulation_steps),
                "optimizer_steps_per_epoch_est": int(opt_steps_per_epoch),
                "planned_optimizer_steps_est": int(planned_opt_steps),
                "global_step_actual": int(gs) if gs is not None else None,
                "train_wall_time_seconds": float(train_seconds),
                "avg_seconds_per_optimizer_step": (float(train_seconds) / float(gs)) if (gs is not None and gs > 0) else None,
                "output_dir": training_args.output_dir,
                "tag": tag,
                "drop_layers": drop_layers,
                "lora_meta": {"target_modules": target_modules, "r": int(lora_r), "alpha": int(lora_alpha), "dropout": float(lora_dropout), "note": "from CLI"},
                "ckpt_path": model_args.pruned_model_path,
                "meta_files": {
                    "run_started_json": paths["run_started_json"],
                    "lora_meta_json": paths["lora_meta_json"],
                    "interrupt_flag": (paths["interrupt_flag"] if interrupted else None),
                },
            }

            safe_write_json(paths["timing_json"], timing)
            print(f"[time] status={timing['status']}  train_wall_time_seconds={train_seconds:.2f}s", flush=True)
            if gs is not None and gs > 0:
                print(f"[time] global_step={gs}  avg_seconds_per_optimizer_step={train_seconds/gs:.4f}s", flush=True)
            print(f"[time] saved timing to: {paths['timing_json']}", flush=True)

if __name__ == "__main__":
    main()

