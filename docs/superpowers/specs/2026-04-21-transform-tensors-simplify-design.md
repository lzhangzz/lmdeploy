# `@transform_tensors` Simplification

## Problem

The current `@transform_tensors` decorator is ~130 lines with three complexity drivers:

1. **Annotation introspection**: `typing.get_type_hints`, `_is_optional_tensor`, `_is_tuple_of_tensors` — fragile, hard to reason about
2. **Dual output paths**: separate branches for single-Tensor vs tuple return
3. **Split input processing**: `linears` and `pass_kwargs` dicts separated at classification time, merged back in the loop

## Design

Replace annotation introspection with runtime `isinstance(val, Linear)` checks, and unify both the input and output processing into single paths.

### Runtime classification (inputs)

Instead of inspecting type annotations to classify parameters, check `isinstance(val, Linear)` at call time. Each arg is either a `Linear` (extract per-kind tensor, handle 1D/2D) or a passthrough value. One loop, one `fn_kwargs` dict — no separate `linears`/`pass_kwargs` dicts.

`None` for optional tensor args (e.g. `gate=None`) flows through naturally: it's not a `Linear`, so the inner function receives `None` directly.

### Uniform output processing

Normalize single-Tensor returns to a 1-element tuple internally. One loop body handles both cases. Unwrap at the end: return `outputs[0]` for single, return the tuple for multi.

### Removed code

- `_is_optional_tensor` helper
- `_is_tuple_of_tensors` helper
- `import typing`, `import types as _bt`
- `tensor_params` / `optional_params` classification lists
- `linears` / `pass_kwargs` separation dicts

### Final decorator

```python
def transform_tensors(fn):
    """Decorator that lifts a tensor-level transform to Linear-level.

    Convention: args that are ``Linear`` instances are treated as tensor
    inputs; all other args pass through unchanged.  Return type is detected
    at runtime: ``Tensor`` -> single ``Linear``, ``tuple`` -> tuple of
    ``Linear`` objects.
    """
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        first = next(v for v in bound.arguments.values()
                     if isinstance(v, Linear))
        out_buckets = None

        for kind in first.tensors:
            was_1d = False
            fn_kwargs = {}

            for name, val in bound.arguments.items():
                if isinstance(val, Linear):
                    t = val.tensors[kind]
                    if t.dim() == 1:
                        was_1d = True
                        t = t.unsqueeze(0)
                    fn_kwargs[name] = t
                else:
                    fn_kwargs[name] = val

            result = fn(**fn_kwargs)
            if not isinstance(result, tuple):
                result = (result,)
            if out_buckets is None:
                out_buckets = [{} for _ in result]
            for i, item in enumerate(result):
                out_buckets[i][kind] = item.squeeze(0) if was_1d else item

        outputs = tuple(
            Linear(ts, weight_format=first.weight_format,
                   data_format=first.data_format)
            for ts in out_buckets)
        return outputs if len(outputs) > 1 else outputs[0]

    return wrapper
```

~45 lines vs ~130 lines. No annotation introspection. One input path. One output path.

## Scope

Only `_base.py` changes. The decorated functions in `attention.py` and the tests are unaffected.
