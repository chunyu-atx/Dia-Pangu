# prune_llm_only_and_merge_CN_cli.py
import os
import re
import json
import argparse
from typing import Dict, List, Optional, Tuple

import torch

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

from transformers import AutoConfig, AutoModelForCausalLM
from msmodelslim import set_logger_level
from msmodelslim.common.prune.transformer_prune.prune_model import PruneConfig, prune_model_weight

# ====== LLM 层前缀/正则（与你的 ckpt keys 一致） ======
LLM_LAYER_PREFIX = "lang_model.base_model.model.model.layers."
LLM_LAYER_PATTERN = r"lang_model\.base_model\.model\.model\.layers\.(\d+)\."

def relax_msmodelslim_size_limit():
    """
    有些环境下 msmodelslim 会对大文件做 size limitation，这里放宽限制（保持你原脚本行为）。
    """
    import ascend_utils.common.security.path as secpath
    import msmodelslim.pytorch.prune.transformer_prune.prune_model_torch as pmt

    orig = secpath.get_valid_read_path

    def relaxed_get_valid_read_path(*args, **kwargs):
        try:
            return orig(*args, **kwargs)
        except ValueError as e:
            msg = str(e)
            if "exceeds size limitation" in msg:
                path = kwargs.get("path", args[0] if len(args) > 0 else None)
                extensions = kwargs.get("extensions", args[1] if len(args) > 1 else None)
                if extensions is not None:
                    ext = os.path.splitext(path)[1].lstrip(".").lower()
                    if ext not in extensions:
                        raise
                return path
            raise

    secpath.get_valid_read_path = relaxed_get_valid_read_path
    pmt.get_valid_read_path = relaxed_get_valid_read_path

def load_checkpoint_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict) and any(hasattr(v, "shape") for v in obj.values()):
        return obj
    if isinstance(obj, dict):
        for k in ["state_dict", "model", "module"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    raise RuntimeError(f"Unrecognized checkpoint format: {type(obj)}")

def infer_llm_num_layers(sd: Dict[str, torch.Tensor]) -> int:
    layer_re = re.compile(re.escape(LLM_LAYER_PREFIX) + r"(\d+)\.")
    max_id = -1
    for k in sd.keys():
        m = layer_re.match(k)
        if m:
            max_id = max(max_id, int(m.group(1)))
    if max_id < 0:
        raise RuntimeError(f"Cannot infer LLM layer count. Expected prefix: {LLM_LAYER_PREFIX}")
    return max_id + 1

def build_layer_id_map_keep_layers(old_n: int, drop_layers: List[int]) -> Dict[int, int]:
    drop = set(drop_layers)
    keep = [i for i in range(old_n) if i not in drop]
    if not keep:
        raise ValueError("After dropping, no layers remain.")
    return {new_i: old_i for new_i, old_i in enumerate(keep)}

def load_adapter_config_if_exists(ckpt_path: str) -> Optional[dict]:
    """
    peft 通常会在同目录保存 adapter_config.json（含 r、lora_alpha、target_modules 等）。
    若存在则优先使用，避免 alpha 推断不准。
    """
    ckpt_dir = os.path.dirname(ckpt_path)
    cand = os.path.join(ckpt_dir, "adapter_config.json")
    if os.path.isfile(cand):
        try:
            with open(cand, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def infer_lora_meta_from_ckpt(sd: Dict[str, torch.Tensor]) -> Tuple[List[str], int]:
    """
    从 keys 推断 LoRA 的 target_modules 和 rank r：
      ...self_attn.q_proj.lora_A.default.weight -> target=q_proj, r=shape[0]
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
        raise RuntimeError("Cannot infer LoRA target_modules: no self_attn.*.lora_A.default.weight found.")
    if len(r_set) != 1:
        raise RuntimeError(f"LoRA rank is inconsistent across modules: {sorted(list(r_set))}")
    return sorted(list(targets)), int(next(iter(r_set)))

def clean_merged_llm_layers_inplace(sd: Dict[str, torch.Tensor], new_layers: int) -> None:
    """
    合并后必须清理掉 layers.{new_layers} 及以上的所有 keys，
    否则会残留原 ckpt 的高层（你遇到的 layer33 问题就是这个）。
    """
    pat = re.compile(r"^" + re.escape(LLM_LAYER_PREFIX) + r"(\d+)\.")
    to_del = []
    max_layer = -1
    for k in list(sd.keys()):
        m = pat.match(k)
        if not m:
            continue
        lid = int(m.group(1))
        max_layer = max(max_layer, lid)
        if lid >= new_layers:
            to_del.append(k)

    print(f"[CLEAN] max layer id found in merged (before clean): {max_layer}")
    print(f"[CLEAN] target new_layers: {new_layers}  (keep 0..{new_layers-1})")
    print(f"[CLEAN] delete keys: {len(to_del)}")

    for k in to_del:
        del sd[k]

    still_bad = any((pat.match(k) and int(pat.match(k).group(1)) >= new_layers) for k in sd.keys())
    print(f"[CLEAN] post-check: any layer id >= new_layers ? {still_bad}")

class LLMOnlyPruneFriendly(torch.nn.Module):
    """
    生成与 ckpt 对齐的 key 结构：
      lang_model.base_model.model.model.layers.*
      且对 target_modules 生成 base_layer + lora_A/B（与你的 ckpt key 一致）
    """
    def __init__(
        self,
        lang_model_path: str,
        new_llm_layers: int,
        device: torch.device,
        target_modules: List[str],
        lora_r: int,
        lora_alpha: int,
    ):
        super().__init__()

        cfg = AutoConfig.from_pretrained(lang_model_path, trust_remote_code=True, local_files_only=True)
        if not hasattr(cfg, "num_hidden_layers"):
            raise AttributeError("Config has no num_hidden_layers; please check your config fields.")
        cfg.num_hidden_layers = new_llm_layers

        base = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)

        from peft import LoraConfig, get_peft_model, TaskType
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            target_modules=target_modules,
            bias="none",
        )
        self.lang_model = get_peft_model(base, lora_cfg)
        self.lang_model.to(device)
        self.lang_model.half()

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

def main():
    parser = argparse.ArgumentParser("LLM-only prune + merge + clean (v0112)")
    parser.add_argument("--ckpt", type=str, required=True, help="Original full checkpoint (pytorch_model.bin)")
    parser.add_argument("--lang-model-path", type=str, required=True, help="Base LLM model directory")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--drop-layers", type=str, required=True, help='Layer spec e.g. "23-28" or "0-2,5,7-9"')

    # 可选：手动覆盖 LoRA（一般不需要，默认从 ckpt/adapter_config.json 推断）
    parser.add_argument("--lora-target-modules", type=str, default="", help='Override like "q_proj,v_proj"')
    parser.add_argument("--lora-r", type=int, default=0, help="Override LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=0, help="Override LoRA alpha")

    parser.add_argument("--tag", type=str, default="", help="Optional output tag override")
    parser.add_argument("--no-clean", action="store_true", help="Do not clean merged checkpoint (NOT recommended)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs if present")
    args = parser.parse_args()

    set_logger_level("info")
    os.makedirs(args.out_dir, exist_ok=True)

    drop_layers = parse_layers_spec(args.drop_layers)
    tag = (args.tag.strip() or make_drop_tag(drop_layers))

    out_llm_only = os.path.join(args.out_dir, f"llm_only_pruned_state_{tag}_CN_cpu.bin")
    out_merged_clean = os.path.join(args.out_dir, f"pytorch_model_pruned_{tag}_merged_clean_CN_cpu.bin")
    out_map = os.path.join(args.out_dir, f"prune_layer_map_{tag}_CN_cpu.json")

    for p in [out_llm_only, out_merged_clean, out_map]:
        if os.path.exists(p) and (not args.overwrite):
            raise FileExistsError(f"Output exists: {p} (use --overwrite to overwrite)")

    print(f"[ARGS] ckpt={args.ckpt}")
    print(f"[ARGS] lang_model_path={args.lang_model_path}")
    print(f"[ARGS] out_dir={args.out_dir}")
    print(f"[ARGS] drop_layers={drop_layers}")
    print(f"[ARGS] tag={tag}")

    sd_orig = load_checkpoint_state_dict(args.ckpt)

    # ===== 推断/确定 LoRA meta =====
    inferred_targets, inferred_r = infer_lora_meta_from_ckpt(sd_orig)
    adapter_cfg = load_adapter_config_if_exists(args.ckpt)

    if args.lora_target_modules.strip():
        target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    else:
        target_modules = adapter_cfg.get("target_modules", inferred_targets) if adapter_cfg else inferred_targets
        if isinstance(target_modules, (set, tuple)):
            target_modules = list(target_modules)
        target_modules = list(target_modules)

    if args.lora_r > 0:
        lora_r = int(args.lora_r)
    else:
        lora_r = int(adapter_cfg.get("r", inferred_r)) if adapter_cfg else inferred_r

    if args.lora_alpha > 0:
        lora_alpha = int(args.lora_alpha)
    else:
        lora_alpha = int(adapter_cfg.get("lora_alpha", 2 * lora_r)) if adapter_cfg else (2 * lora_r)

    print(f"[LoRA] target_modules={target_modules}, r={lora_r}, alpha={lora_alpha}, adapter_config_used={adapter_cfg is not None}")

    old_n = infer_llm_num_layers(sd_orig)
    print(f"[INFO] inferred original LLM layers: {old_n}")

    bad = [x for x in drop_layers if x < 0 or x >= old_n]
    if bad:
        raise ValueError(f"drop_layers out of range [0,{old_n-1}]: {bad}")

    layer_id_map = build_layer_id_map_keep_layers(old_n, drop_layers)
    new_n = len(layer_id_map)
    print(f"[INFO] new LLM layers after drop: {new_n} (keep 0..{new_n-1})")

    device = torch.device("cpu")
    model_small = LLMOnlyPruneFriendly(
        lang_model_path=args.lang_model_path,
        new_llm_layers=new_n,
        device=device,
        target_modules=target_modules,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    )

    prune_config = (
        PruneConfig()
        .set_steps(["prune_blocks", "prune_bert_intra_block"])
        .add_blocks_params(LLM_LAYER_PATTERN, layer_id_map)
    )

    relax_msmodelslim_size_limit()
    prune_model_weight(model_small, prune_config, weight_file_path=args.ckpt)

    # 保存 LLM-only pruned state
    sd_llm_pruned = model_small.state_dict()
    torch.save(sd_llm_pruned, out_llm_only)
    print(f"[OK] Saved LLM-only pruned state: {out_llm_only}")

    # 合并 + （可选）清理
    merged = dict(sd_orig)
    for k, v in sd_llm_pruned.items():
        merged[k] = v

    if not args.no_clean:
        clean_merged_llm_layers_inplace(merged, new_layers=new_n)
    else:
        print("[WARN] --no-clean is set. You may get leftover higher layers in merged checkpoint!")

    torch.save(merged, out_merged_clean)
    print(f"[OK] Saved merged checkpoint: {out_merged_clean}")

    with open(out_map, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tag": tag,
                "old_layers": old_n,
                "new_layers": new_n,
                "drop_layers": drop_layers,
                "layer_id_map": layer_id_map,
                "llm_layer_prefix": LLM_LAYER_PREFIX,
                "llm_layer_pattern": LLM_LAYER_PATTERN,
                "device": "cpu",
                "ckpt": args.ckpt,
                "lang_model_path": args.lang_model_path,
                "lora_target_modules": target_modules,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "adapter_config_used": adapter_cfg is not None,
                "note": "If clean is enabled, layers >= new_layers are removed from merged checkpoint.",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[OK] Saved prune map: {out_map}")
    print(f"[REMIND] When loading for inference/training, set config.num_hidden_layers == {new_n} (or use drop_layers to rebuild).")

if __name__ == "__main__":
    main()

