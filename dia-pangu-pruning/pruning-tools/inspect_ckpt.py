import os, re, torch
from collections import Counter

# CKPT = "/media/t1/zcy/dia-pangu/checkpoint_new/dia-pangu_v1219/pytorch_model.bin"  # 修改为要检查的权重路径
# CKPT = "/media/t1/sym/workspace_prune_new/ft_out_v0112_drop23_28/F_c23_28_P1_FLLM_v0112.bin"  # 修改为要检查的权重路径
CKPT = "/media/t1/shujie/dia-pangu/checkpoint_EN/P1-FLLM_en/pytorch_model.cpu_state_dict.bin"  # 修改为要检查的权重路径
# CKPT = "/media/t1/sym/workspace_prune_new/output/pytorch_model_pruned_drop23_28_clean_lite_v19.bin"  # 修改为要检查的权重路径
obj = torch.load(CKPT, map_location="cpu")
if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()) and any(hasattr(v, "shape") for v in obj.values()):
    sd = obj
else:
    # 常见包裹形式
    for k in ["state_dict", "model", "module"]:
        if isinstance(obj, dict) and k in obj and isinstance(obj[k], dict):
            sd = obj[k]
            break
    else:
        raise RuntimeError(f"Unrecognized ckpt format: {type(obj)} keys={list(obj.keys())[:20] if isinstance(obj, dict) else None}")

print("Total keys:", len(sd))
print("Example keys:")
for k in list(sd.keys())[:30]:
    v = sd[k]
    shp = tuple(v.shape) if hasattr(v, "shape") else None
    print(" ", k, shp)

# 统计“带层号”的前缀候选：xxxx.<digit>.
pat = re.compile(r"^(.*?)(\d+)\.")
cnt = Counter()
for k in sd.keys():
    m = pat.search(k)
    if m:
        prefix = m.group(1)  # 含最后一个点
        cnt[prefix] += 1

print("\nTop layer-prefix candidates (prefix -> count):")
for p, c in cnt.most_common(20):
    print(f"{p} -> {c}")

# 猜一个“最像 transformer blocks”的候选：出现次数最多且包含 layer/block/blocks/h/layers
hints = ["layer", "layers", "block", "blocks", ".h.", "decoder", "transformer"]
cand = [(p, c) for p, c in cnt.items() if any(h in p for h in hints)]
cand.sort(key=lambda x: x[1], reverse=True)
print("\nHeuristic transformer-like prefixes:")
for p, c in cand[:20]:
    print(f"{p} -> {c}")

