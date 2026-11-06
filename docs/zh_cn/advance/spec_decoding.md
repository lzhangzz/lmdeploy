# 投机解码

投机解码是一种优化技术，它通过引入轻量级草稿模型来预测多个后续 token，再由主模型在一次前向推理中验证并选择匹配度最高的最长 token 序列。与标准的自回归解码相比，这种方法可使系统一次性生成多个 token。

:::{note}
请注意，这是 lmdeploy 中的实验性功能。
:::

TurboMind 引擎支持以下投机解码方法：

| 方法     | 目标模型                     | 草稿权重                                |
| -------- | ---------------------------- | --------------------------------------- |
| `mtp`    | 带 MTP 层的 Qwen3.5 系列模型 | 目标模型 checkpoint 中的多 token 预测层 |
| `eagle3` | Qwen3 系列模型               | 独立的 EAGLE-3 草稿模型（`model`）      |

## 示例

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

## 限制

启用投机解码时，结构化输出（`response_format`，参见[结构化输出](./structed_output.md)）不会生效：TurboMind 的投机验证路径不会应用语法掩码，受约束的请求会生成不受约束的文本。需要结构化输出的请求请关闭投机解码。
