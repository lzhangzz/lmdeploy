# Speculative Decoding

Speculative decoding is an optimization technique that introduces a lightweight draft model to propose multiple next tokens. The main model then verifies them and keeps the longest matched prefix in a single forward pass. Compared with standard auto-regressive decoding, this lets the system generate multiple tokens at once.

:::{note}
This is an experimental feature in lmdeploy.
:::

The TurboMind engine supports the following speculative methods:

| method   | target models                  | draft weights                                         |
| -------- | ------------------------------ | ----------------------------------------------------- |
| `mtp`    | Qwen3.5 family with MTP layers | the target checkpoint's multi-token-prediction layers |
| `eagle3` | Qwen3 family                   | a separate EAGLE-3 draft checkpoint (`model`)         |

## Examples

### MTP

#### MTP pipeline

```python
from lmdeploy import TurbomindEngineConfig, pipeline
from lmdeploy.messages import SpeculativeConfig


if __name__ == '__main__':

    spec_cfg = SpeculativeConfig(method='mtp', num_speculative_tokens=3)
    pipe = pipeline('Qwen/Qwen3.5-35B-A3B',
                    backend_config=TurbomindEngineConfig(tp=2, max_batch_size=128),
                    speculative_config=spec_cfg)
    response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
    print(response)
```

#### MTP serving

```shell
lmdeploy serve api_server \
Qwen/Qwen3.5-35B-A3B \
--tp 2 \
--server-port 24545 \
--speculative-algorithm mtp \
--speculative-num-draft-tokens 3 \
--max-batch-size 128
```

### Eagle 3

#### Eagle 3 pipeline

```python
from lmdeploy import TurbomindEngineConfig, pipeline
from lmdeploy.messages import SpeculativeConfig


if __name__ == '__main__':

    spec_cfg = SpeculativeConfig(
        method='eagle3',
        num_speculative_tokens=3,
        model='/path/to/qwen3-eagle3-draft',
    )
    pipe = pipeline('Qwen/Qwen3-8B',
                    backend_config=TurbomindEngineConfig(max_batch_size=128),
                    speculative_config=spec_cfg)
    response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
    print(response)
```

#### Eagle 3 serving

```shell
lmdeploy serve api_server \
Qwen/Qwen3-8B \
--server-port 24545 \
--speculative-algorithm eagle3 \
--speculative-draft-model /path/to/qwen3-eagle3-draft \
--speculative-num-draft-tokens 3 \
--max-batch-size 128
```

## Limitations

Structured output (`response_format`, see [structured output](./structed_output.md)) is not applied when speculative decoding is enabled: TurboMind's speculative verification path does not apply grammar masks, so constrained requests generate unconstrained text. Disable speculative decoding for requests that need structured output.
