# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import torch

from ..linear import Linear
from ._base import Builder, SplitSide

# ---------------------------------------------------------------------------
# TP split rules for MLA projections
# ---------------------------------------------------------------------------

_MLA_TP_RULES: dict[str, dict] = {
    "q_a_proj":  dict(split_side=SplitSide.OUTPUT),
    "q_b_proj":  dict(split_side=SplitSide.OUTPUT),
    "kv_a_proj": dict(split_side=SplitSide.OUTPUT),
    "wo":        dict(split_side=SplitSide.INPUT),
}


# ---------------------------------------------------------------------------
# MLABuilder -- MLA projections, fold+pad, norms
# ---------------------------------------------------------------------------


class MLABuilder(Builder):
    """MLA (Multi-head Latent Attention) weight loading builder."""

    def add_projections(self, *, q_a_proj, q_b_proj, kv_a_proj, kv_b_proj,
                        wo):
        """Apply MLA fold+pad, then commit each projection.

        The fold consumes kv_b_proj -- its information is absorbed into
        q_b_proj and wo.  After the fold, kv_b_proj is not committed.
        """
        linears = {
            "q_a_proj": q_a_proj,
            "q_b_proj": q_b_proj,
            "kv_a_proj": kv_a_proj,
            "wo": wo,
        }
        if kv_b_proj is not None:
            linears["kv_b_proj"] = kv_b_proj

        self._fold_and_pad(linears)

        model_dtype = self.config.data_type
        for name, lin in linears.items():
            if lin is None:
                continue
            rule = _MLA_TP_RULES.get(name, {})
            split_side = rule.get('split_side')
            self._commit_linear(name, lin, split_side=split_side,
                                model_dtype=model_dtype)

    def add_norms(self, *, q_a_norm, kv_a_norm, data_type=None):
        """Create norm children for q_a_layernorm and kv_a_layernorm."""
        if q_a_norm is not None:
            self._add_norm_child('q_a_layernorm', q_a_norm,
                                 data_type=data_type)
        if kv_a_norm is not None:
            self._add_norm_child('kv_a_layernorm', kv_a_norm,
                                 data_type=data_type)

    # ------------------------------------------------------------------
    # MLA fold+pad (moved from glm4_moe_lite_spec)
    # ------------------------------------------------------------------

    def _fold_and_pad(self, linears: dict[str, Linear]):
        """Fold kv_b_proj into q_b_proj and wo, then pad wo.

        Weight tensors are temporarily transposed to HF layout [out, in]
        for the fold arithmetic, then transposed back to TM layout [in, out].
        """
        # Temporarily convert weight tensors from TM [in, out] to HF [out, in].
        for lin in linears.values():
            for k in list(lin.tensors.keys()):
                t = lin.tensors[k]
                if t.dim() >= 2:
                    lin.tensors[k] = t.t().contiguous()
        try:
            self._fold_and_pad_hf(linears)
        finally:
            # Convert weight tensors back from HF [out, in] to TM [in, out].
            for lin in linears.values():
                for k in list(lin.tensors.keys()):
                    t = lin.tensors[k]
                    if t.dim() >= 2:
                        lin.tensors[k] = t.t().contiguous()

    def _fold_and_pad_hf(self, linears: dict[str, Linear]):
        """Inner fold logic; expects all weight tensors in HF layout [out, in]."""
        cfg = self.config
        head_num = cfg.head_num
        qk_rope_dim = cfg.qk_rope_dim
        qk_nope_dim = cfg.qk_nope_dim
        kv_lora_rank = cfg.kv_lora_rank
        v_head_dim = cfg.v_head_dim
        size_per_head = cfg.head_dim

        q_b_lin = linears.get("q_b_proj")
        kv_b_lin = linears.pop("kv_b_proj", None)
        o_lin = linears.get("wo")

        if q_b_lin is not None and kv_b_lin is not None and o_lin is not None:
            q_b = q_b_lin.tensors.get("weight")
            kv_b = kv_b_lin.tensors.get("weight")
            o = o_lin.tensors.get("weight")

            if (q_b is not None and kv_b is not None and o is not None
                    and torch.is_floating_point(q_b)
                    and torch.is_floating_point(kv_b)):
                orig_q_head_dim = q_b.size(0) // head_num
                orig_qk_nope_dim = orig_q_head_dim - qk_rope_dim
                orig_v_head_dim = o.size(1) // head_num
                target_nope_dim = size_per_head - qk_rope_dim

                if (orig_qk_nope_dim != target_nope_dim
                        or orig_v_head_dim != v_head_dim):
                    # Split kv_b into kc and vc
                    kv_b_per_head = kv_b.reshape(
                        head_num, orig_qk_nope_dim + orig_v_head_dim,
                        kv_lora_rank)
                    kc_w = kv_b_per_head[:, :orig_qk_nope_dim, :]
                    vc_w = kv_b_per_head[:, orig_qk_nope_dim:, :]

                    # Fold kc into q_b_proj
                    q_b_per_head = q_b.reshape(
                        head_num, orig_q_head_dim, q_b.size(1))
                    q_nope_w = q_b_per_head[:, :orig_qk_nope_dim, :]
                    q_rope_w = q_b_per_head[:, orig_qk_nope_dim:, :]
                    q_nope_expanded = torch.bmm(
                        kc_w.transpose(1, 2), q_nope_w)
                    q_b_folded = torch.cat(
                        [q_nope_expanded, q_rope_w], dim=1)
                    q_b_lin.tensors["weight"] = q_b_folded.reshape(
                        head_num * size_per_head, q_b.size(1))

                    # Fold vc into o_proj
                    o_per_head = o.reshape(
                        o.size(0), head_num, orig_v_head_dim)
                    o_folded = torch.bmm(
                        o_per_head.permute(1, 0, 2), vc_w)
                    o_lin.tensors["weight"] = o_folded.permute(
                        1, 0, 2).reshape(
                            o.size(0), head_num * kv_lora_rank)

        # Pad wo from [hidden, head_num*v_head_dim]
        #           to [hidden, head_num*size_per_head]
        if o_lin is not None:
            o_w = o_lin.tensors["weight"]
            cur_v = o_w.size(1) // head_num
            if cur_v < size_per_head:
                o_w = o_w.reshape(o_w.size(0), head_num, cur_v)
                o_w = torch.nn.functional.pad(
                    o_w, (size_per_head - cur_v, 0, 0, 0, 0, 0))
                o_lin.tensors["weight"] = o_w.reshape(
                    o_w.size(0), head_num * size_per_head)
