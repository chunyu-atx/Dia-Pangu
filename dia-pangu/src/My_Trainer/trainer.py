import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.distributed as dist
from transformers import TrainingArguments
import numpy as np
import random
import os

# 初始化 HCCL（NPU DDP）后端
if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="hccl")

# 可选：DataCollator 类型
from typing import Callable, Optional, Sequence, Dict

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        local_rank: int,
        train_dataset=None,
        eval_dataset=None,
        data_collator: Optional[Callable] = None,
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.local_rank = local_rank

        # 设置设备
        torch.npu.set_device(self.local_rank)
        self.device = torch.device(f"npu:{self.local_rank}")
        self.model.to(self.device)

        # Mixed precision
        self.use_bf16 = getattr(args, "bf16", True)
        self.scaler = torch.cpu.amp.GradScaler() if not self.use_bf16 else None

        # 优化器
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=getattr(args, "learning_rate", 5e-5),
        )

        # Batch size
        self.train_batch_size = getattr(args, "batch_size_3D", 1)
        self.eval_batch_size = getattr(args, "batch_size_3D", 1)

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=getattr(self.args, "dataloader_num_workers", 4),
            pin_memory=True,
        )

    def get_eval_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=getattr(self.args, "dataloader_num_workers", 4),
            pin_memory=True,
        )

    def train(self, epochs=1):
        self.model.train()
        train_loader = self.get_train_dataloader()

        for epoch in range(epochs):
            for step, batch in enumerate(train_loader):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                self.optimizer.zero_grad()
                if self.use_bf16:
                    with torch.amp.autocast(device_type='npu', dtype=torch.bfloat16):
                        outputs = self.model(**batch)
                        loss = outputs["loss"]
                    loss.backward()
                else:
                    outputs = self.model(**batch)
                    loss = outputs["loss"]
                    loss.backward()

                self.optimizer.step()
                if step % 10 == 0:
                    print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")

    def evaluate(self):
        self.model.eval()
        eval_loader = self.get_eval_dataloader()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                preds = outputs["logits"].argmax(-1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(batch["labels"].cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        acc = (all_preds == all_labels).mean()
        print(f"Eval Accuracy: {acc:.4f}")
        return acc

    def save(self, output_dir):
        torch.save(self.model.state_dict(), f"{output_dir}/pytorch_model.bin")
        print(f"Model saved to {output_dir}/pytorch_model.bin")
