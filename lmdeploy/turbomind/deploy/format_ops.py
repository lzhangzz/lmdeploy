# Copyright (c) OpenMMLab. All rights reserved.
"""Operation validity checkers for quantized tensor bundles.

These functions read `DataFormat` fields (exposed via pybind11 from C++)
and determine whether dimension-level operations are safe on quantized
tensor bundles.  They are decision predicates used for branching -- not
exception guards.
"""

from __future__ import annotations

from _turbomind import DataFormat
from _turbomind import QuantParamDesc


def can_permute(fmt: DataFormat, dim0: int, dim1: int) -> bool:
    """Can we permute dim0 <-> dim1 without breaking block structure?

    Returns False if either dimension has block_size > 1, because permuting
    would cross block boundaries and invalidate quantization parameter
    alignment.
    """
    if dim0 >= len(fmt.block_sizes) or dim1 >= len(fmt.block_sizes):
        return True
    return fmt.block_sizes[dim0] == 1 and fmt.block_sizes[dim1] == 1


def can_split(fmt: DataFormat, dim: int) -> bool:
    """Can we split along *dim* without breaking block structure?

    Requires block_size == 1 along that dimension.
    """
    if dim >= len(fmt.block_sizes):
        return True
    return fmt.block_sizes[dim] == 1


def param_shape(data_shape: list[int], block_sizes: list[int],
                param: QuantParamDesc) -> list[int]:
    """Compute quant param shape given data shape and block sizes.

    Accounts for reduced dimensions (ceil-div by block_size) and
    transposed layout.
    """
    shape: list[int] = []
    for i, s in enumerate(data_shape):
        bs = block_sizes[i] if i < len(block_sizes) else 1
        if bs > 1:
            shape.append(-(-s // bs))  # ceil div
        else:
            shape.append(s)
    if param.transposed:
        shape.reverse()
    return shape
