# Copyright (c) OpenMMLab. All rights reserved.
"""Optional W4A16 backend using the TurboMind Linear API."""

import torch

import lmdeploy.pytorch.distributed as dist
from lmdeploy.turbomind.linear import Linear, get_engine
from lmdeploy.turbomind.weight_format import AWQFormat

from ..awq_modules import LinearW4A16Impl


class TurbomindAwqLinearW4A16Impl(LinearW4A16Impl):
    """Adapt canonical AWQ parameters to the TurboMind Linear API."""

    def __init__(self, in_features: int, out_features: int, w_bit: int, group_size: int):
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self._linear: Linear | None = None

    def _release(self):
        linear, self._linear = self._linear, None
        if linear is None:
            return
        # Forward is asynchronous and may use any engine stream. Release the
        # prepared weight only after all device work completes.
        torch.cuda.synchronize()
        linear.close()

    def __del__(self):
        try:
            linear = self._linear
            if linear is None:
                return
            with torch.cuda.device(linear.device):
                self._release()
        except Exception:
            pass

    def update_weights(
        self, qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor, bias: torch.Tensor | None = None
    ):
        self._release()

        plan = get_engine().get_weight_plan(weight_format=AWQFormat(block_in=self.group_size), dtype=scales.dtype)
        linear = Linear(plan=plan)
        linear.prepare_weight(qweight, scales=scales, zeros=qzeros)
        # Weight preparation is asynchronous on the loading stream, while the
        # engine executes on another stream. Publish only after packing completes.
        torch.cuda.current_stream().synchronize()

        self._linear = linear
        return qweight, scales, qzeros, bias

    def forward(
        self,
        x,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        bias: torch.Tensor | None = None,
        all_reduce: bool = False,
        group: torch.distributed.ProcessGroup | None = None,
    ):
        linear = self._linear

        exec_plan = linear.get_exec_plan(x)
        out, _ = linear(x, exec_plan=exec_plan)

        if bias is not None:
            out = out + bias

        if out.ndim == 2:
            out = out.unsqueeze(0)
        if all_reduce:
            dist.all_reduce(out, group=group)
        return out
