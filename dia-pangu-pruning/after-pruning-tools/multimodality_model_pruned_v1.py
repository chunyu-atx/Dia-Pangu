import copy
import re
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from peft import get_peft_model, LoraConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .my_embedding_layer import MyEmbedding


# ===== v0112：只处理语言模型层（非常关键，避免误伤视觉侧的 ".layers."）=====
LLM_LAYER_PREFIX = "lang_model.base_model.model.model.layers."
LLM_LAYER_RE = re.compile(r"^" + re.escape(LLM_LAYER_PREFIX) + r"(\d+)\.")


def get_rank0() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _get_num_layers(cfg) -> int:
    for name in ("num_hidden_layers", "n_layer", "num_layers", "num_layer"):
        if hasattr(cfg, name):
            return int(getattr(cfg, name))
    raise ValueError(f"Cannot find num-layers field in config: {cfg.__class__.__name__}")


def _set_num_layers(cfg, n: int) -> None:
    found = False
    for name in ("num_hidden_layers", "n_layer", "num_layers", "num_layer"):
        if hasattr(cfg, name):
            setattr(cfg, name, int(n))
            found = True
    if not found:
        raise ValueError(f"Cannot set num-layers field in config: {cfg.__class__.__name__}")


def _collect_llm_layer_indices(state_dict_keys) -> List[int]:
    idxs = set()
    for k in state_dict_keys:
        m = LLM_LAYER_RE.match(k)
        if m:
            idxs.add(int(m.group(1)))
    return sorted(idxs)


def _infer_expected_llm_layers_from_model(model: nn.Module) -> Optional[int]:
    """
    从 model.state_dict() 里扫 lang_model.base_model.model.model.layers.<id>.* 来推断层数。
    """
    idxs = _collect_llm_layer_indices(model.state_dict().keys())
    if not idxs:
        return None
    return max(idxs) + 1


def remap_state_dict_drop_layers_v0112(
    state_dict: dict,
    drop_layers: List[int],
    expected_llm_layers: Optional[int] = None,
) -> dict:
    """
    v0112 安全 remap：
    - 仅 remap LLM 层前缀 lang_model.base_model.model.model.layers.*
    - 若 ckpt 已经是连续索引 0..expected_llm_layers-1，则直接返回（跳过 remap）
    - 否则认为 ckpt 仍保留原索引（比如 0..33 但缺 23..28），按 drop_layers 重排
    """
    if not drop_layers:
        return state_dict

    layer_idxs = _collect_llm_layer_indices(state_dict.keys())
    if not layer_idxs:
        # 没有 LLM 层 key，直接返回
        return state_dict

    # ✅ 已经是 pruned 连续索引：直接跳过 remap（这是适配你 v0112 “合并清理后 ckpt” 的关键）
    if expected_llm_layers is not None and set(layer_idxs) == set(range(expected_llm_layers)):
        return state_dict

    # 推断“原始编号空间”的大小（仅在“仍保留原索引”的情况下成立）
    orig_layers = max(layer_idxs) + 1

    drop = set(int(x) for x in drop_layers if 0 <= int(x) < orig_layers)
    if not drop:
        # drop_layers 全都不在 ckpt 的层编号空间里 -> 不 remap
        return state_dict

    keep_layers = [i for i in range(orig_layers) if i not in drop]
    new_layers = len(keep_layers)

    # 如果 ckpt 本身已经是 0..new_layers-1 连续索引，也直接返回
    if set(layer_idxs) == set(range(new_layers)):
        return state_dict

    old2new = {}
    new_i = 0
    for i in range(orig_layers):
        if i in drop:
            continue
        old2new[i] = new_i
        new_i += 1

    def _remap_key(k: str) -> Optional[str]:
        m = LLM_LAYER_RE.match(k)
        if not m:
            return k  # 非 LLM key 不动
        old = int(m.group(1))
        if old in drop:
            return None
        if old in old2new:
            new = old2new[old]
            # 把 layers.<old>. 替换成 layers.<new>.
            # 这里用 match 的 span 精确替换数字部分
            start = m.start(1)
            end = m.end(1)
            return k[:start] + str(new) + k[end:]
        return k

    out = {}
    for k, v in state_dict.items():
        nk = _remap_key(k)
        if nk is None:
            continue
        out[nk] = v
    return out


def smart_load_state_dict(model: nn.Module, ckpt: dict, drop_layers: List[int]) -> None:
    """
    v0112 安全加载：
    - 去 module. 前缀
    - 仅对 LLM 层做“必要时的 remap”
    - 只加载 key 存在且 shape 一致的权重
    """
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        ckpt = ckpt["state_dict"]

    ckpt = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt.items()}

    expected_llm_layers = _infer_expected_llm_layers_from_model(model)
    ckpt = remap_state_dict_drop_layers_v0112(ckpt, drop_layers, expected_llm_layers=expected_llm_layers)

    cur = model.state_dict()
    filtered = {}
    mismatch = []
    for k, v in ckpt.items():
        if k in cur and hasattr(v, "shape") and hasattr(cur[k], "shape") and v.shape == cur[k].shape:
            filtered[k] = v
        elif k in cur:
            mismatch.append((k, tuple(getattr(v, "shape", ())), tuple(getattr(cur[k], "shape", ()))))

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    if get_rank0() == 0:
        print(
            f"[ckpt] loaded_keys={len(filtered)}  missing={len(missing)}  "
            f"unexpected={len(unexpected)}  mismatch={len(mismatch)}"
        )
        if mismatch[:20]:
            print("[ckpt] first mismatches (up to 20):")
            for k, s1, s2 in mismatch[:20]:
                print(f"  - {k}: ckpt={s1} model={s2}")


class MultiPanguForCausalLMPruned(nn.Module):
    def __init__(
        self,
        text_tokenizer_path: str,
        lang_model_path: str,
        drop_layers: Optional[List[int]] = None,
        device: Optional[torch.device] = None,
        use_bf16: bool = True,
        # ✅ v0112：LoRA 参数可从外部传入（train 脚本会从 ckpt / adapter_config.json 推断）
        lora_target_modules: Optional[List[str]] = None,
        lora_r: int = 4,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        drop_layers = drop_layers or []
        self.drop_layers = drop_layers

        # tokenizer：保持与你们原版一致
        self.image_padding_tokens = []
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            text_tokenizer_path,
            use_fast=False,
            trust_remote_code=True,
            local_files_only=True,
        )
        special_token = {"additional_special_tokens": ["<image>", "</image>"]}
        max_img_size = 100
        image_num = 32
        for i in range(max_img_size):
            image_padding_token = ""
            for j in range(image_num):
                image_token = "<image" + str(i * image_num + j) + ">"
                image_padding_token += image_token
                special_token["additional_special_tokens"].append(image_token)
            self.image_padding_tokens.append(image_padding_token)

        self.text_tokenizer.add_special_tokens(special_token)
        self.text_tokenizer.pad_token_id = 0
        self.text_tokenizer.bos_token_id = 1
        self.text_tokenizer.eos_token_id = 45892

        # ===== 按 drop_layers 重建 pruned config =====
        cfg = AutoConfig.from_pretrained(lang_model_path, trust_remote_code=True, local_files_only=True)
        orig_layers = _get_num_layers(cfg)
        keep_layers = [i for i in range(orig_layers) if i not in set(drop_layers)]
        pruned_layers = len(keep_layers)

        cfg_pruned = copy.deepcopy(cfg)
        _set_num_layers(cfg_pruned, pruned_layers)

        # 用 from_config：场景二（外部 load_state_dict）
        self.lang_model = AutoModelForCausalLM.from_config(cfg_pruned, trust_remote_code=True)

        if device is None:
            device = torch.device(
                "npu" if torch.npu.is_available()
                else "cuda" if torch.cuda.is_available()
                else "cpu"
            )
        self.lang_model.to(device)

        dtype = torch.bfloat16 if use_bf16 else torch.float16
        if dtype != torch.float32:
            self.lang_model.to(dtype)

        # ===== LoRA（必须先 wrap，再 load_state_dict，才能对齐 v0112 的 base_layer / lora_A/B keys）=====
        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "v_proj"]

        lora_config = LoraConfig(
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            target_modules=list(lora_target_modules),
            lora_dropout=float(lora_dropout),
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.lang_model = get_peft_model(self.lang_model, lora_config)

        if get_rank0() == 0:
            print("[lora] Trainable parameters after applying LoRA:")
            self.lang_model.print_trainable_parameters()

        self.lang_model.gradient_checkpointing_enable()
        self.lang_model.enable_input_require_grads()

        self.embedding_layer = MyEmbedding()
        self.embedding_layer.weight = self.lang_model.get_input_embeddings().weight
        self.loss_function = nn.CrossEntropyLoss()

        self.hidden_dim = 4096
        self.voc_size = 153376

    def forward(self, lang_x, vision_x, cls_labels, attention_mask, labels, loss_reweight, key_words_query):
        B = vision_x.shape[0]
        input_embedding, sim = self.embedding_layer(vision_x, cls_labels, lang_x, key_words_query, mode="train")
        output = self.lang_model(inputs_embeds=input_embedding, attention_mask=attention_mask, labels=labels)

        targets = torch.zeros(B * 14, device=sim.device)
        ctr_loss = F.cross_entropy(sim, targets.long())

        if get_rank0() == 0:
            print("lm_loss:", float(output["loss"]), "ctr_loss:", float(ctr_loss))

        return {"loss": output["loss"] + ctr_loss * 4}

