from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import torch

from .benchmark import (
    BenchmarkRequest,
    BenchmarkTask,
    apply_cp_pattern,
    diff_metrics,
)
from .cases import InputTensors, RunCase


class FlashQlaUnsupported(ValueError):
    pass


@dataclass(frozen=True)
class FlashQlaResult:
    out: torch.Tensor | None
    final_state: torch.Tensor | None
    cp_enabled: bool


@cache
def _imports() -> dict[str, object]:
    from flash_qla.ops.gated_delta_rule import chunk as qla_chunk
    from flash_qla.ops.gated_delta_rule.chunk.cp import build_cp_context, cp_preprocess_fwd
    from flash_qla.ops.utils import chunk_local_cumsum

    return {
        'chunk_size': qla_chunk.CHUNK_SIZE,
        'chunk_local_cumsum': chunk_local_cumsum,
        'kkt_solve': qla_chunk.kkt_solve,
        'fused_gdr_fwd': qla_chunk.fused_gdr_fwd,
        'build_cp_context': build_cp_context,
        'cp_preprocess_fwd': cp_preprocess_fwd,
    }


def is_available() -> bool:
    try:
        _imports()
    except (ImportError, OSError, RuntimeError, ValueError):
        return False
    return True


def _cp_context(k, v, cu_seqlens):
    imports = _imports()
    if k.shape[0] == 1 and cu_seqlens is None:
        cu_seqlens = torch.tensor([0, k.shape[1]], device=k.device, dtype=torch.int32)
    return imports['build_cp_context'](
        cu_seqlens, enable_intra=k.shape[0] == 1, num_v_heads=v.shape[2],
        chunk_size=imports['chunk_size'], is_train=False,
    )


def all_forward(q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                g: torch.Tensor,
                beta: torch.Tensor,
                *,
                initial_state: torch.Tensor | None = None,
                cu_seqlens: torch.Tensor | None = None,
                cp_level: str = 'all') -> FlashQlaResult:
    imports = _imports()
    if cp_level not in ('all', 'off'):
        raise FlashQlaUnsupported(f'unsupported_cp_level {cp_level}')
    context = _cp_context(k, v, cu_seqlens) if cp_level == 'all' else None
    raw_offsets = context.cu_seqlens if context is not None else cu_seqlens
    a = imports['kkt_solve'](k=k, b=beta, cu_seqlens=raw_offsets, chunk_size=imports['chunk_size'])
    g_cumsum = imports['chunk_local_cumsum'](
        g, cu_seqlens=raw_offsets, chunk_size=imports['chunk_size'],
    )
    cp_enabled = context is not None and context.is_intra
    if cp_enabled:
        initial_state, _ = imports['cp_preprocess_fwd'](
            context, k=k, v=v, a=a, g=g_cumsum, beta=beta, initial_state=initial_state,
            state_v_first=False, enable_fwd_cp_cache=False,
        )
    out, _h, final_state = imports['fused_gdr_fwd'](
        q=q, k=k, v=v, a=a, g=g_cumsum, b=beta, initial_state=initial_state,
        output_final_state=True, output_h=False, output_o=True,
        cu_seqlens=context.intra_cp_cu_seqlens if cp_enabled else raw_offsets,
        cp_seq_map=context.seq_map_c2r if cp_enabled else None,
        raw_cu_seqlens=raw_offsets if cp_enabled else None, state_v_first=False,
    )
    return FlashQlaResult(out=out, final_state=final_state, cp_enabled=cp_enabled)


def _cp_pattern_segment_tokens(inputs: InputTensors) -> int:
    context = _cp_context(inputs.k, inputs.v, inputs.offsets)
    if not context.is_intra:
        raise ValueError('cp_not_selected')
    cp_seq_map = context.seq_map_c2r
    if cp_seq_map.unique().numel() == cp_seq_map.numel():
        raise ValueError('cp_pattern_requires_multiple_segments')
    offsets = context.intra_cp_cu_seqlens
    return int((offsets[1:] - offsets[:-1]).max().item())


def make_benchmark_task(run: RunCase, inputs: InputTensors, request: BenchmarkRequest,
                        device: torch.device) -> BenchmarkTask:
    from .benchmark import base_row
    from .reference import chunk_gated_delta_rule_fwd

    if request.cp_pattern != 'auto':
        if request.cp_level != 'all':
            raise ValueError('cp_not_selected')
        inputs = apply_cp_pattern(
            run.input,
            inputs,
            request.cp_pattern,
            _cp_pattern_segment_tokens(inputs),
        )
    row = base_row(
        run,
        'flashqla',
        cp_level=request.cp_level,
        cp_pattern=request.cp_pattern,
    )

    def execute():
        result = all_forward(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.g,
            inputs.beta,
            initial_state=inputs.h0,
            cu_seqlens=inputs.offsets,
            cp_level=request.cp_level,
        )
        row['cp_enabled'] = result.cp_enabled
        return result

    def validate():
        actual = execute()
        expected_o, expected_state = chunk_gated_delta_rule_fwd(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.g,
            inputs.beta,
            initial_state=inputs.h0,
            cu_seqlens=inputs.offsets,
            chunk_size=run.chunk_size,
        )
        if request.validate_outputs:
            torch.testing.assert_close(actual.out, expected_o, rtol=8e-2, atol=8e-2)
            if actual.final_state is not None or expected_state is not None:
                torch.testing.assert_close(actual.final_state, expected_state, rtol=8e-2, atol=8e-2)
        return {
            **diff_metrics('output', actual.out, expected_o),
            **diff_metrics('state', actual.final_state, expected_state),
        }

    return BenchmarkTask(row, execute, validate=validate)
