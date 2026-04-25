# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import torch

from ..linear import Linear
from ._base import Builder, SplitSide

# ---------------------------------------------------------------------------
# MLA fold+pad pipeline (standalone functions)
# ---------------------------------------------------------------------------


def fold_kv_b(q_b: Linear, kv_b: Linear, wo: Linear, *,
              cfg) -> tuple[Linear, Linear]:
    """Fold kv_b into q_b and wo. Returns (q_b_folded, wo_folded).

    Splits kv_b into key-compressed (kc) and value-compressed (vc) parts.
    Folds kc into q_b via matmul (q_nope @ kc^T per head).
    Folds vc into wo via matmul (vc @ wo per head).
    All arithmetic in TM layout [in, out].
    """
    head_num = cfg.head_num
    qk_rope_dim = cfg.qk_rope_dim
    size_per_head = cfg.head_dim

    q_b_w = q_b.tensors["weight"]
    kv_b_w = kv_b.tensors["weight"]
    o_w = wo.tensors["weight"]

    # Derive original dimensions from tensor shapes
    orig_q_head_dim = q_b_w.shape[-1] // head_num
    orig_qk_nope_dim = orig_q_head_dim - qk_rope_dim
    orig_v_head_dim = o_w.shape[0] // head_num

    # Split kv_b into kc and vc: [kv_lora_rank, head_num, dim]
    kv_b_h = kv_b_w.reshape(kv_b_w.shape[0], head_num, -1)
    kc = kv_b_h[:, :, :orig_qk_nope_dim]
    vc = kv_b_h[:, :, orig_qk_nope_dim:]

    # Fold kc into q_b: q_nope @ kc^T per head
    q_b_h = q_b_w.reshape(q_b_w.shape[0], head_num, orig_q_head_dim)
    q_nope = q_b_h[:, :, :orig_qk_nope_dim].permute(1, 0, 2)   # [H, R, P]
    q_rope = q_b_h[:, :, orig_qk_nope_dim:].permute(1, 0, 2)   # [H, R, S]
    kc_t = kc.permute(1, 2, 0)                                  # [H, P, R]
    q_expanded = torch.bmm(q_nope, kc_t)                        # [H, R, R]
    q_folded = torch.cat([q_expanded, q_rope], dim=-1)          # [H, R, sp]
    q_folded = q_folded.permute(1, 0, 2).reshape(
        q_b_w.shape[0], head_num * size_per_head)

    # Fold vc into wo: vc @ wo per head
    vc_b = vc.permute(1, 0, 2)                                  # [H, R, V]
    o_h = o_w.reshape(head_num, orig_v_head_dim, -1)            # [H, V, N]
    o_folded = torch.bmm(vc_b, o_h)                             # [H, R, N]
    o_folded = o_folded.reshape(head_num * o_folded.shape[1], -1)

    return (Linear(tensors={"weight": q_folded.contiguous()},
                   weight_format=q_b.weight_format,
                   data_format=q_b.data_format),
            Linear(tensors={"weight": o_folded.contiguous()},
                   weight_format=wo.weight_format,
                   data_format=wo.data_format))


def pad_wo_input(wo: Linear, *, cfg) -> Linear:
    """Pad wo input dim from head_num * cur_dim to head_num * size_per_head."""
    head_num = cfg.head_num
    size_per_head = cfg.head_dim
    w = wo.tensors["weight"]
    cur_dim = w.shape[0] // head_num
    w = w.reshape(head_num, cur_dim, -1)
    w = torch.nn.functional.pad(w, (0, 0, size_per_head - cur_dim, 0))
    w = w.reshape(head_num * size_per_head, -1)
    return Linear(tensors={"weight": w.contiguous()},
                  weight_format=wo.weight_format,
                  data_format=wo.data_format)


# ---------------------------------------------------------------------------
# MLABuilder -- MLA projections, fold+pad, norms
# ---------------------------------------------------------------------------


class MLABuilder(Builder):
    """MLA (Multi-head Latent Attention) weight loading builder."""

    def add_projections(self, *, q_a_proj, q_b_proj, kv_a_proj, kv_b_proj,
                        wo):
        """Apply MLA fold+pad, then commit each projection."""
        q_b_proj, wo = fold_kv_b(q_b_proj, kv_b_proj, wo, cfg=self.config)
        wo = pad_wo_input(wo, cfg=self.config)

        for name, lin, side in [
            ("q_a_proj", q_a_proj, None),
            ("q_b_proj", q_b_proj, SplitSide.OUTPUT),
            ("kv_a_proj", kv_a_proj, None),
            ("wo", wo, SplitSide.INPUT),
        ]:
            self._add_linear(name, lin, split_side=side)

