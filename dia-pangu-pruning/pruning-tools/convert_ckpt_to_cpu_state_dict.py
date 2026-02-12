# convert_ckpt_to_cpu_state_dict.py
import os
import torch

SRC = "/media/t1/shujie/dia-pangu/checkpoint_EN/P1-FLLM_en/pytorch_model.bin"
DST = "/media/t1/shujie/dia-pangu/checkpoint_EN/P1-FLLM_en/pytorch_model.cpu_state_dict.bin"

obj = torch.load(SRC, map_location="cpu")

# 提取 state_dict
sd = None
if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()) and any(hasattr(v, "shape") for v in obj.values()):
    sd = obj
elif isinstance(obj, dict):
    for k in ["state_dict", "model", "module"]:
        if k in obj and isinstance(obj[k], dict):
            sd = obj[k]
            break
if sd is None:
    raise RuntimeError(f"Unrecognized ckpt format: {type(obj)}")

# 全量转 CPU + contiguous（尽量去掉任何 NPU 私货）
sd_cpu = {}
for k, v in sd.items():
    if torch.is_tensor(v):
        sd_cpu[k] = v.detach().cpu().contiguous()
    else:
        sd_cpu[k] = v

torch.save(sd_cpu, DST)
print("Saved:", DST, "keys:", len(sd_cpu))

