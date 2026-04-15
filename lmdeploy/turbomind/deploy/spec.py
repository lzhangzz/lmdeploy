# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
import torch
from .linear import Linear
from .module_configs import SpecAttnConfig

if TYPE_CHECKING:
    from .builder import SplitSide


# ===================================================================
# New pipeline: TextModelSpec + composable-ops TextModelLoader
# ===================================================================


class TextModelSpec(ABC):
    """Declarative weight mapping for a model architecture.

    Subclasses define how to read and transform weights for a specific model.
    Methods return ``Linear`` for linear layers and raw ``Tensor`` for norms,
    embeddings, scalars, etc.

    The ``TextModelLoader`` consumes a spec: it iterates the returned dicts,
    applies TP split rules, and commits each weight to C++.
    """

    params: dict[str, torch.Tensor]

    # Default values for configure() fields; overwritten by configure().
    _attn_tp: int = 1
    _permute_qk: bool = True
    _repeat_kv: int = 0
    _head_dim: int = 0
    _rope_dim: int = 0
    _attn_output_gate: bool = False
    _kv_head_num: int = 0
    # TODO: there dont belong here
    _linear_qkv_split: tuple[int, int, int] | None = None
    _gdn_qkv_split: tuple[int, int, int] | None = None

    # --- Builder-driven loading context (injected by TextModelLoader) ---
    _contexts: list = None  # GPU contexts
    _root_handles: list = None  # Root C++ ModelWeight handles

    def model(self):
        """Build the full model hierarchy using builders.

        Override in subclasses.
        """

    # -- Configuration (called by TextModelLoader before processing) --

    def configure(self, cfg: SpecAttnConfig):
        """Set TP and model parameters needed for QKV merge and GDN fusion.

        Args:
            cfg: SpecAttnConfig with tp, permute_qk, repeat_kv, etc.

        Called by ``TextModelLoader`` before processing each layer batch.
        Idempotent — safe to call repeatedly with the same values.
        """
        self._attn_tp = cfg.tp
        self._permute_qk = cfg.permute_qk
        self._repeat_kv = cfg.repeat_kv
        self._head_dim = cfg.head_dim
        self._rope_dim = cfg.rope_dim if cfg.rope_dim else cfg.head_dim
        self._attn_output_gate = cfg.output_gate
        self._kv_head_num = cfg.kv_head_num

    # -- Common helpers (subclasses may override) --

    def _get(self, key: str) -> torch.Tensor | None:
        """Get a raw tensor from the checkpoint params."""
        return self.params.get(key)

    def _read_linear(self, prefix: str) -> Linear | None:
        """Read a Linear bundle from the checkpoint at *prefix*.

        Probes all known suffixes and auto-detects the format via
        ``WeightFormat.accepts``.  Override for model-specific logic.
        """
        from .kind_map import build_linear
        return build_linear(self.params, prefix)

    _FFN_MAP = [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]

    def _read_ffn_linears(self, pfx: str) -> dict[str, Linear]:
        """Read standard gate/down/up FFN linears."""
        result: dict[str, Linear] = {}
        for tm_name, hf_key in self._FFN_MAP:
            lin = self._read_linear(f"{pfx}.{hf_key}")
            if lin is not None:
                result[tm_name] = lin
        return result

    # -- Linear bundles (TP-split by the transformer) --

    def attn_linears(self, layer: int) -> dict[str, Linear]:
        """Return ``{tm_name: Linear}`` for attention weights.

        Wraps ``_read_attn_linears()``.  If the raw dict contains the
        ``w_qkv.{q,k,v}`` pattern, merges them into a single ``w_qkv``
        ``Linear`` (already TP-interleaved and in TM layout).
        MLA specs return non-QKV keys and the merge is skipped.
        """
        raw = self._read_attn_linears(layer)
        if "w_qkv.q" in raw and "w_qkv.k" in raw and "w_qkv.v" in raw:
            from .builder import merge_qkv_linear
            q = raw.pop("w_qkv.q")
            k = raw.pop("w_qkv.k")
            v = raw.pop("w_qkv.v")
            merged = merge_qkv_linear(
                q, k, v,
                tp=self._attn_tp,
                head_dim=self._head_dim,
                rope_dim=self._rope_dim,
                permute_qk=self._permute_qk,
                attn_output_gate=self._attn_output_gate,
                repeat_kv=self._repeat_kv,
                kv_head_num=self._kv_head_num,
            )
            raw["w_qkv"] = merged
            # Pad wo bias with zeros if QKV has bias but wo does not
            o_lin = raw.get("wo")
            if o_lin is not None and "bias" in merged.tensors and "bias" not in o_lin.tensors:
                o_lin.tensors["bias"] = torch.zeros_like(merged.tensors["bias"])
        return raw

    def _read_attn_linears(self, layer: int) -> dict[str, Linear]:
        """Override in subclasses to return raw attention ``Linear`` bundles."""
        return {}

    def ffn_linears(self, layer: int) -> dict[str, Linear]:
        """Return ``{tm_name: Linear}`` for dense FFN / shared-expert weights."""
        return {}

    def moe_ffn_linears(self, layer: int, expert: int) -> dict[str, Linear]:
        """Return ``{tm_name: Linear}`` for one MoE routed expert."""
        return {}

    def linear_attn_linears(self, layer: int) -> dict[str, Linear]:
        """Return ``{tm_name: Linear}`` for linear-attention (GDN) weights.

        Wraps ``_read_linear_attn_linears()``.  If the raw dict contains any
        of the GDN in-proj keys, fuses them into ``in_proj_all`` with TP
        interleaving via ``fuse_gdn_in_proj()``.
        """
        raw = self._read_linear_attn_linears(layer)
        from .builder import _GDN_IN_PROJ_KEYS, fuse_gdn_in_proj
        if any(k in raw for k in _GDN_IN_PROJ_KEYS):
            raw = fuse_gdn_in_proj(raw, self._attn_tp,
                                   qkv_split=self._linear_qkv_split)
        return raw

    def _read_linear_attn_linears(self, layer: int) -> dict[str, Linear]:
        """Override in subclasses to return raw GDN ``Linear`` bundles."""
        return {}

    # -- Raw tensors (broadcast or simple split) --

    def attn_norm(self, layer: int) -> torch.Tensor | None:
        return None

    def ffn_norm(self, layer: int) -> torch.Tensor | None:
        return None

    def tok_embeddings(self) -> torch.Tensor | None:
        return None

    def output_weight(self) -> torch.Tensor | None:
        return None

    def norm_weight(self) -> torch.Tensor | None:
        return None

    def attn_params(
        self, layer: int
    ) -> dict[str, tuple[torch.Tensor, SplitSide | None]]:
        """Return direct attention parameters.

        Each entry is ``{param_name: (tensor, split_side)}``.
        ``param_name`` is a single-segment name within the attention subtree
        (e.g. ``"sinks"``).
        ``split_side`` is ``None`` for broadcast, or a ``SplitSide`` value.
        """
        return {}

    def attn_norm_children(self, layer: int) -> dict[str, torch.Tensor]:
        """Return norm submodule weights for the attention block.

        Each entry is ``{child_name: tensor}``.  The loader creates a
        ``NormConfig`` child and commits the tensor as ``"weight"``.
        All norm children are broadcast (no TP split).
        """
        return {}

    def moe_params(
        self, layer: int
    ) -> dict[str, tuple[torch.Tensor, SplitSide | None]]:
        """Return non-expert MoE parameters.

        Each entry is ``{param_name: (tensor, split_side)}``.
        ``param_name`` is the leaf name within the moe_ffn subtree
        (e.g. ``"gate.weight"``, ``"score_correction_bias"``).
        """
        return {}

    def moe_gate(
        self, layer: int
    ) -> dict[str, 'Linear']:
        """Return MoE gate and shared_gate as Linear bundles.

        Keys are child module names (e.g. ``"gate"``, ``"shared_gate"``).
        Values are :class:`Linear` bundles with weight (and optional bias).
        """
        return {}

    def linear_attn_params(
        self, layer: int
    ) -> dict[str, tuple[torch.Tensor, SplitSide | None]]:
        """Return direct linear-attention (GDN) parameters.

        Each entry is ``{param_name: (tensor, split_side)}``.
        ``param_name`` is a single-segment name within the linear_attn
        subtree (e.g. ``"A_log"``, ``"dt_bias"``, ``"conv1d"``).
        """
        return {}

    def linear_attn_norm_children(self, layer: int) -> dict[str, torch.Tensor]:
        """Return norm submodule weights for the linear-attention block.

        Each entry is ``{child_name: tensor}``.  The loader creates a
        ``NormConfig`` child and commits the tensor as ``"weight"``.
        All norm children are broadcast (no TP split).
        """
        return {}

    def _permute_qk_tensors(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE layout permutation to Q and K norm weights.

        Uses the ``_permute_qk``, ``_head_dim``, and ``_rope_dim`` fields
        set by ``configure()``.
        """
        if not self._permute_qk:
            return q, k
        from .builder import _reorder_rotary_emb
        q = _reorder_rotary_emb(q, self._head_dim, self._rope_dim)
        k = _reorder_rotary_emb(k, self._head_dim, self._rope_dim)
        return q, k

    # -- metadata --

    @abstractmethod
    def model_info(self) -> dict:
        """Return model metadata (num_layer, head_num, etc.)."""

    def num_experts(self, layer: int) -> int:
        return 0
