# layer_mask.py
import torch
import torch.nn as nn


class ProjHeadMaskWrapper(nn.Module):
    """
    Wrap a projection Linear layer (q_proj/k_proj/v_proj) and apply head mask on its output.

    IMPORTANT:
    - For q_proj: out_features = num_heads * head_dim
    - For k_proj/v_proj in GQA/MQA: out_features = num_key_value_heads * head_dim
    """
    def __init__(self, base_linear: nn.Module, head_dim: int, num_out_heads: int = None):
        super().__init__()
        self.base_linear = base_linear
        self.head_dim = int(head_dim)

        # num_out_heads can be inferred if not provided
        self.num_out_heads = None if num_out_heads is None else int(num_out_heads)

        device = base_linear.weight.device
        dtype = base_linear.weight.dtype

        if self.num_out_heads is not None:
            self.register_buffer("head_mask", torch.ones(self.num_out_heads, device=device, dtype=dtype))
            self._all_one = True
        else:
            # will be initialized on first forward (lazy)
            self.register_buffer("head_mask", torch.empty(0, device=device, dtype=dtype))
            self._all_one = True

    def _lazy_init_mask(self, out_dim: int):
        # out_dim should be num_out_heads * head_dim
        assert out_dim % self.head_dim == 0, f"out_dim={out_dim} not divisible by head_dim={self.head_dim}"
        self.num_out_heads = out_dim // self.head_dim
        self.head_mask = torch.ones(self.num_out_heads, device=self.head_mask.device, dtype=self.head_mask.dtype)
        self._all_one = True

    @torch.no_grad()
    def set_all_zero(self):
        if self.num_out_heads is None:
            # not initialized yet; mark as not all one; actual mask will be created on first forward
            self._all_one = False
            return
        self.head_mask = torch.zeros_like(self.head_mask)
        self._all_one = False

    @torch.no_grad()
    def set_all_one(self):
        if self.num_out_heads is None:
            self._all_one = True
            return
        self.head_mask = torch.ones_like(self.head_mask)
        self._all_one = True

    def forward(self, x):
        y = self.base_linear(x)  # [..., out_dim]
        out_dim = y.shape[-1]

        if self.num_out_heads is None:
            self._lazy_init_mask(out_dim)

        if self._all_one:
            return y

        prefix = y.shape[:-1]
        # reshape to [..., num_out_heads, head_dim]
        y2 = y.reshape(*prefix, self.num_out_heads, self.head_dim)

        view_shape = [1] * len(prefix) + [self.num_out_heads, 1]
        y2 = y2 * self.head_mask.view(*view_shape)
        return y2.reshape(*prefix, -1)


def mask_attention_layer(model, layer_idx: int, mode: str = "v_only"):
    """
    Mask the whole attention layer at layer_idx by forcing V (and optionally Q) to zero.

    mode:
      - "v_only": mask v_proj output heads (KV heads) to 0 -> attention output becomes 0
      - "qv": additionally mask q_proj output heads (Q heads) to 0

    Returns:
      - enable_mask()
      - disable_mask()
      - restore()
    """
    attn = model.model.layers[layer_idx].self_attn

    # Q heads and KV heads (GQA/MQA)
    num_q_heads = int(getattr(attn, "num_heads"))
    num_kv_heads = int(getattr(attn, "num_key_value_heads", num_q_heads))
    head_dim = int(getattr(attn, "head_dim"))

    original_q = attn.q_proj
    original_v = attn.v_proj

    # v_proj uses KV heads count
    v_wrap = ProjHeadMaskWrapper(original_v, head_dim=head_dim, num_out_heads=num_kv_heads)
    v_wrap.to(original_v.weight.device)
    attn.v_proj = v_wrap

    q_wrap = None
    if mode == "qv":
        # q_proj uses Q heads count
        q_wrap = ProjHeadMaskWrapper(original_q, head_dim=head_dim, num_out_heads=num_q_heads)
        q_wrap.to(original_q.weight.device)
        attn.q_proj = q_wrap

    def enable_mask():
        attn.v_proj.set_all_zero()
        if q_wrap is not None:
            attn.q_proj.set_all_zero()

    def disable_mask():
        attn.v_proj.set_all_one()
        if q_wrap is not None:
            attn.q_proj.set_all_one()

    def restore():
        attn.v_proj = original_v
        if q_wrap is not None:
            attn.q_proj = original_q

    return enable_mask, disable_mask, restore
