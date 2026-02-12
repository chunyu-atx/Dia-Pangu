import tqdm.auto as tqdm
import transformers
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List
import torch
import numpy as np
import random
import os
import re
import csv
import json
import time
import statistics

from torch.utils.data import DataLoader
from Dataset.dataset_test import dataset_test
from Model.multimodality_model_pruned_v1 import MultiPanguForCausalLM

# ========= 关键：提前 patch accelerate，避免 peft import 时 ImportError =========
def patch_accelerate_clear_device_cache():
    """
    peft 的 loftq_utils 会: from accelerate.utils.memory import clear_device_cache
    某些 accelerate 版本没有该函数，导致 ImportError。
    这里提前往 accelerate.utils.memory 注入同名函数，使 peft 能正常导入。
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

from peft import LoraConfig, get_peft_model, TaskType


# ======== 与微调后 ckpt keys 对齐用 ========
# 你的 key 示例：lang_model.base_model.model.model.layers.0....
LLM_LAYER_PREFIX = "lang_model.base_model.model.model.layers."
LLM_LAYER_RE = re.compile(re.escape(LLM_LAYER_PREFIX) + r"(\d+)\.")


def setup_seed(seed: int):
    torch.manual_seed(seed)
    # 若无 NPU 环境，这行会报错；你这里显然是 NPU 环境，就保留
    torch.npu.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(20)


def npu_sync_if_available():
    # NPU 推理默认异步，为了计时准确，计时前后都 synchronize
    try:
        if hasattr(torch, "npu"):
            torch.npu.synchronize()
    except Exception:
        pass


def resolve_local_model_dir(path: str) -> str:
    """
    transformers 会把“不存在的本地路径”当成 hub repo id 去校验，从而报错
    所以必须确保传进去的是“真实存在的目录”。
    """
    if os.path.isdir(path):
        return path

    alt = path.replace("_", "-")
    if os.path.isdir(alt):
        print(f"[WARN] tokenizer/model dir not found: {path}")
        print(f"[WARN] fallback to existing dir:  {alt}")
        return alt

    raise FileNotFoundError(
        f"Local model/tokenizer directory not found.\n"
        f"  tried: {path}\n"
        f"  tried: {alt}\n"
        f"Please check the real directory name on your server."
    )


def load_state_dict_any(ckpt_path: str, map_location="cpu") -> Dict[str, torch.Tensor]:
    obj = torch.load(ckpt_path, map_location=map_location)
    # 兼容两种格式：直接就是 state_dict，或包在 state_dict/model/module 下
    if isinstance(obj, dict) and any(hasattr(v, "shape") for v in obj.values()):
        return obj
    if isinstance(obj, dict):
        for k in ["state_dict", "model", "module"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    raise RuntimeError(f"Unrecognized checkpoint format: {type(obj)}")


def infer_llm_num_layers_from_ckpt(sd: Dict[str, torch.Tensor]) -> int:
    max_id = -1
    for k in sd.keys():
        m = LLM_LAYER_RE.match(k)
        if m:
            max_id = max(max_id, int(m.group(1)))
    if max_id < 0:
        raise RuntimeError(f"Cannot infer LLM layer count. Expected prefix: {LLM_LAYER_PREFIX}")
    return max_id + 1


def scan_llm_layer_ids(sd: Dict[str, torch.Tensor]) -> List[int]:
    ids = set()
    for k in sd.keys():
        m = LLM_LAYER_RE.match(k)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def find_min_layer_ge(layer_ids: List[int], threshold: int) -> Optional[int]:
    for x in layer_ids:
        if x >= threshold:
            return x
    return None


def ckpt_has_lora(sd: Dict[str, torch.Tensor]) -> bool:
    # 你的微调 ckpt key 里有：...lora_A.default.weight / ...lora_B.default.weight
    for k in sd.keys():
        if ".lora_A." in k or ".lora_B." in k:
            return True
    return False


def infer_lora_meta_from_ckpt(sd: Dict[str, torch.Tensor]) -> Tuple[List[str], int]:
    """
    从 ckpt keys 推断：
      ...self_attn.q_proj.lora_A.default.weight (r, hidden)
      ...self_attn.v_proj.lora_A.default.weight (r, hidden)
    => target_modules=['q_proj','v_proj'], r=...
    """
    targets = set()
    r_set = set()
    pat = re.compile(r"self_attn\.([a-zA-Z0-9_]+)\.lora_A\.default\.weight$")

    for k, v in sd.items():
        m = pat.search(k)
        if m and hasattr(v, "shape"):
            targets.add(m.group(1))
            r_set.add(int(v.shape[0]))

    if not targets:
        raise RuntimeError("Cannot infer LoRA target_modules: no self_attn.*.lora_A.default.weight found in ckpt.")
    if len(r_set) != 1:
        raise RuntimeError(f"LoRA rank inconsistent across modules: {sorted(list(r_set))}")

    return sorted(list(targets)), int(next(iter(r_set)))


def wrap_lang_model_with_lora(m, target_modules: List[str], lora_r: int, lora_alpha: int):
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
    )
    # 关键：必须 wrap，才能匹配你 ckpt 里的 base_layer / lora_A.default / lora_B.default 这些 key
    m.lang_model = get_peft_model(m.lang_model, lora_cfg)
    return m


def safe_percentile(vals: List[float], p: float) -> Optional[float]:
    if not vals:
        return None
    vals_sorted = sorted(vals)
    k = int(round((p / 100.0) * (len(vals_sorted) - 1)))
    k = max(0, min(k, len(vals_sorted) - 1))
    return vals_sorted[k]


@dataclass
class ModelArguments:
    lang_encoder_path: Optional[str] = field(default="/media/t1/zcy/openPangu_Embedded_7B_V1_1")
    tokenizer_path: str = field(default="/media/t1/zcy/openPangu_Embedded_7B_V1_1")

    # ✅ 微调后模型（你重命名后的文件）
    ckpt_path: str = field(
        default="/media/t1/sym/workspace_prune_new/ft_out_v0112_drop23_28/F_c23_28_P1_FLLM_v0112.bin"
    )

    out_dir: str = field(default="/media/t1/sym/dia-pangu/results/")
    out_name: str = field(default="dia-pangu_v0112_ft_c23_28.csv")

    npu_id: int = field(default=0)

    # ✅ LoRA（默认与你给的 keys 对齐：q_proj/v_proj, r=4, alpha=8）
    # 如果你想仍然“自动从 ckpt 推断”，可以把 use_lora_meta_from_ckpt=True
    use_lora_meta_from_ckpt: bool = field(default=False)
    lora_target_modules: str = field(default="q_proj,v_proj")  # 逗号分隔
    lora_r: int = field(default=4)
    lora_alpha: int = field(default=8)


@dataclass
class DataArguments:
    Mode: Optional[str] = field(default="Train")
    test_split: Optional[str] = field(default="open")


def main():
    overall_t0 = time.perf_counter()

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    # ✅ 选择 NPU
    device = torch.device(f"npu:{model_args.npu_id}")
    torch.npu.set_device(device)
    print(f"[INFO] Using device: {device}")

    # ✅ 确保 tokenizer/model 目录真实存在
    model_args.tokenizer_path = resolve_local_model_dir(model_args.tokenizer_path)
    model_args.lang_encoder_path = resolve_local_model_dir(model_args.lang_encoder_path)

    # ====== Data ======
    print("Setup Data")
    Test_dataset = dataset_test(text_tokenizer=model_args.tokenizer_path)
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

    # ====== Load ckpt (CPU) ======
    t_ckpt0 = time.perf_counter()
    print(f"[INFO] Loading ckpt to CPU: {model_args.ckpt_path}")
    ckpt_sd_cpu = load_state_dict_any(model_args.ckpt_path, map_location="cpu")
    ckpt_load_time_s = time.perf_counter() - t_ckpt0

    # ====== 推断层数 ======
    t_meta0 = time.perf_counter()
    layer_ids = scan_llm_layer_ids(ckpt_sd_cpu)
    inferred_layers = infer_llm_num_layers_from_ckpt(ckpt_sd_cpu)
    num_hidden_layers = inferred_layers

    # ====== LoRA meta（与你给的 keys 对齐） ======
    has_lora = ckpt_has_lora(ckpt_sd_cpu)
    if has_lora:
        if model_args.use_lora_meta_from_ckpt:
            inferred_targets, inferred_r = infer_lora_meta_from_ckpt(ckpt_sd_cpu)
            target_modules = inferred_targets
            lora_r = inferred_r
            # 没有 adapter_config 的情况下，alpha 兜底：2*r
            lora_alpha = 2 * lora_r
            print(f"[INFO] LoRA meta inferred from ckpt: target_modules={target_modules}, r={lora_r}, alpha={lora_alpha}")
        else:
            target_modules = [x.strip() for x in model_args.lora_target_modules.split(",") if x.strip()]
            lora_r = int(model_args.lora_r)
            lora_alpha = int(model_args.lora_alpha)
            print(f"[INFO] LoRA meta from args: target_modules={target_modules}, r={lora_r}, alpha={lora_alpha}")
    else:
        target_modules, lora_r, lora_alpha = [], 0, 0
        print("[INFO] No LoRA keys detected in ckpt -> will NOT wrap peft LoRA.")

    meta_infer_time_s = time.perf_counter() - t_meta0

    print(f"[INFO] inferred num_hidden_layers from ckpt keys: {inferred_layers}")
    print(f"[INFO] final num_hidden_layers used to build model: {num_hidden_layers}")

    offending = find_min_layer_ge(layer_ids, num_hidden_layers)
    if offending is not None:
        print(f"[WARN] Detected layer {offending} (>= {num_hidden_layers}) keys in ckpt -> ckpt might be NOT clean!")

    # ====== Build model ======
    t_build0 = time.perf_counter()
    model = MultiPanguForCausalLM(
        text_tokenizer_path=model_args.tokenizer_path,
        lang_model_path=model_args.lang_encoder_path,
        num_hidden_layers=num_hidden_layers,
        device_map=None,
        enable_gradient_checkpointing=False,
    )
    model_build_time_s = time.perf_counter() - t_build0

    # ====== ✅ 关键：若 ckpt 含 LoRA，则必须先 wrap，再 load_state_dict ======
    t_lora0 = time.perf_counter()
    if has_lora:
        model = wrap_lang_model_with_lora(model, target_modules=target_modules, lora_r=lora_r, lora_alpha=lora_alpha)
    lora_wrap_time_s = time.perf_counter() - t_lora0

    # Load state_dict on CPU
    t_sd0 = time.perf_counter()
    print("[INFO] Loading state_dict into model (CPU) ...")
    try:
        model.load_state_dict(ckpt_sd_cpu, strict=True)
        strict_used = True
    except RuntimeError as e:
        # 如果你 ckpt 里有少量无关 key（比如保存时多带了点东西），这里会报错
        # 为了让你能顺利跑测试，这里做一次降级并打印差异
        print("[WARN] strict=True load_state_dict failed. Fallback to strict=False.")
        print(f"[WARN] Error: {e}")
        missing, unexpected = model.load_state_dict(ckpt_sd_cpu, strict=False)
        print(f"[WARN] Missing keys: {len(missing)}")
        print(f"[WARN] Unexpected keys: {len(unexpected)}")
        strict_used = False
    state_dict_load_time_s = time.perf_counter() - t_sd0

    # Move to NPU
    t_to0 = time.perf_counter()
    print("[INFO] Moving model to NPU and set eval ...")
    model = model.half().to(device)
    npu_sync_if_available()
    model.eval()
    move_to_npu_time_s = time.perf_counter() - t_to0

    model_load_total_time_s = time.perf_counter() - overall_t0

    # ====== Output files ======
    os.makedirs(model_args.out_dir, exist_ok=True)

    out_csv = os.path.join(model_args.out_dir, model_args.out_name)
    base, _ = os.path.splitext(model_args.out_name)

    per_sample_timing_csv = os.path.join(model_args.out_dir, f"{base}_per_sample_timing.csv")
    timing_json = os.path.join(model_args.out_dir, f"{base}_timing.json")

    # ====== Inference + timing ======
    per_sample_times_ms: List[float] = []
    infer_t0 = time.perf_counter()

    with torch.inference_mode():
        with open(out_csv, mode="w") as outfile, open(per_sample_timing_csv, mode="w") as tfile:
            writer = csv.writer(outfile)
            twriter = csv.writer(tfile)

            writer.writerow(["ID", "Question", "Ground Truth", "Pred", "GT_labels"])
            twriter.writerow(["ID", "InferTimeMs"])

            for sample in tqdm.tqdm(Test_dataloader):
                image_id = int(sample["id"][0])
                question = sample["question"]
                guide = sample["guide"][0]
                cls_labels = sample["cls_labels"][0]

                vision_x = sample["vision_x"].to(device).half()
                answer = sample["answer"][0]

                npu_sync_if_available()
                t_one0 = time.perf_counter()
                question_out, pred = model.generate(question, guide, vision_x)
                npu_sync_if_available()
                dt_ms = (time.perf_counter() - t_one0) * 1000.0
                per_sample_times_ms.append(dt_ms)

                print("finding_gt: ", answer)
                print("finding_pred: ", pred)

                writer.writerow([image_id, question_out, answer, pred, cls_labels])
                twriter.writerow([image_id, f"{dt_ms:.3f}"])

    infer_total_time_s = time.perf_counter() - infer_t0

    # ====== Save timing summary ======
    if per_sample_times_ms:
        avg_ms = float(statistics.mean(per_sample_times_ms))
        p50 = safe_percentile(per_sample_times_ms, 50)
        p90 = safe_percentile(per_sample_times_ms, 90)
        p95 = safe_percentile(per_sample_times_ms, 95)
        p99 = safe_percentile(per_sample_times_ms, 99)
        max_ms = float(max(per_sample_times_ms))
        min_ms = float(min(per_sample_times_ms))
    else:
        avg_ms = p50 = p90 = p95 = p99 = max_ms = min_ms = None

    summary = {
        "device": str(device),
        "npu_id": int(model_args.npu_id),
        "ckpt_path": model_args.ckpt_path,
        "tokenizer_path": model_args.tokenizer_path,
        "lang_encoder_path": model_args.lang_encoder_path,
        "num_hidden_layers": {
            "used": int(num_hidden_layers),
            "inferred_from_ckpt_keys": int(inferred_layers),
        },
        "ckpt_has_lora": bool(has_lora),
        "lora": {
            "target_modules": target_modules,
            "r": int(lora_r),
            "alpha": int(lora_alpha),
        },
        "state_dict_load": {
            "strict_used": bool(strict_used),
        },
        "time_seconds": {
            "ckpt_load_cpu": float(ckpt_load_time_s),
            "meta_infer": float(meta_infer_time_s),
            "model_build": float(model_build_time_s),
            "lora_wrap": float(lora_wrap_time_s),
            "state_dict_load_cpu": float(state_dict_load_time_s),
            "move_to_npu": float(move_to_npu_time_s),
            "model_load_total": float(model_load_total_time_s),
            "inference_total": float(infer_total_time_s),
        },
        "per_sample_infer_ms": {
            "count": int(len(per_sample_times_ms)),
            "avg": avg_ms,
            "p50": p50,
            "p90": p90,
            "p95": p95,
            "p99": p99,
            "min": min_ms,
            "max": max_ms,
        },
        "outputs": {
            "result_csv": out_csv,
            "per_sample_timing_csv": per_sample_timing_csv,
            "timing_json": timing_json,
        },
    }

    with open(timing_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] saved result csv to: {out_csv}")
    print(f"[DONE] saved per-sample timing csv to: {per_sample_timing_csv}")
    print(f"[DONE] saved timing json to: {timing_json}")
    print(f"[INFO] model_load_total_time_s = {model_load_total_time_s:.3f}s, inference_total_time_s = {infer_total_time_s:.3f}s")


if __name__ == "__main__":
    main()