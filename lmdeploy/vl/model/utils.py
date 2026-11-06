# Copyright (c) OpenMMLab. All rights reserved.

import inspect
from collections.abc import Callable, MutableSequence


def _set_func(origin_func_path: str | None, rewrite_func: Callable, origin_func: Callable = None):
    """Replace old function with the new function.

    Args:
        origin_func_path (str): original function path
        rewrite_func (Callable): function to replace with
        origin_func (Callable): function to replace
    """
    # import module
    if isinstance(origin_func_path, str):
        split_path = origin_func_path.split('.')
        for i in range(len(split_path), 0, -1):
            try:
                exec('import {}'.format('.'.join(split_path[:i])))
                break
            except Exception:
                continue

        origin_func = eval(origin_func_path) \
            if origin_func is None else origin_func

    method_class = inspect.ismethod(origin_func)

    # replace method
    if not method_class:
        import gc
        refs = gc.get_referrers(origin_func)
        obj_id = id(origin_func)
        for ref in refs:
            if isinstance(ref, dict):
                for x, y in ref.items():
                    if id(y) == obj_id:
                        ref[x] = rewrite_func
            elif isinstance(ref, MutableSequence):
                for i, v in enumerate(ref):
                    if id(v) == obj_id:
                        ref[i] = rewrite_func
    if isinstance(origin_func_path, str):
        exec(f'{origin_func_path} = rewrite_func')
    elif method_class:
        raise NotImplementedError

    return origin_func
