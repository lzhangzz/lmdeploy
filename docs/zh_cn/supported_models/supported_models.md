# 支持的模型

以下列表为 LMDeploy TurboMind 引擎在 NVIDIA CUDA 平台上支持的模型。架构不在列表中的模型无法使用 LMDeploy 推理。

|              Model               |       Size       | Type | FP16/BF16 | KV INT8 | KV INT4 | W4A16 |
| :------------------------------: | :--------------: | :--: | :-------: | :-----: | :-----: | :---: |
|              Llama               |     7B - 65B     | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|              Llama2              |     7B - 70B     | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|              Llama3              |     8B, 70B      | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|             Llama3.1             |     8B, 70B      | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|     Llama3.2<sup>\[2\]</sup>     |      1B, 3B      | LLM  |    Yes    |  Yes\*  |  Yes\*  |  Yes  |
|            Code Llama            |     7B - 34B     | LLM  |    Yes    |   Yes   |   Yes   |  No   |
|                YI                |     6B - 34B     | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|            InternLM2             |     7B - 20B     | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|           InternLM2.5            |        7B        | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|            InternLM3             |        8B        | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|            Intern-S1             |       241B       | MLLM |    Yes    |   Yes   |   Yes   |  No   |
|          Intern-S1-mini          |       8.3B       | MLLM |    Yes    |   Yes   |   Yes   |  No   |
|        Intern-S2-Preview         |  35B-A3B, 397B   | MLLM |    Yes    |    -    |    -    |   -   |
|         Intern-S2-Mobius         |       35B        | MLLM |    Yes    |    -    |    -    |   -   |
|     Qwen1.5<sup>\[1\]</sup>      |   1.8B - 110B    | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|      Qwen2<sup>\[2\]</sup>       |    0.5B - 72B    | LLM  |    Yes    |  Yes\*  |  Yes\*  |  Yes  |
|            Qwen2-MoE             |     57BA14B      | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|     Qwen2.5<sup>\[2\]</sup>      |    0.5B - 72B    | LLM  |    Yes    |  Yes\*  |  Yes\*  |  Yes  |
|              Qwen3               |    0.6B-235B     | LLM  |    Yes    |   Yes   |  Yes\*  | Yes\* |
|             Qwen3.5              |    0.8B-397B     | MLLM |    Yes    |   Yes   |   No    |  Yes  |
|             Qwen2-VL             |      2B, 7B      | MLLM |    Yes    |    -    |    -    |   -   |
|            Qwen2.5-VL            |     3B - 72B     | MLLM |    Yes    |    -    |    -    |   -   |
|             Mixtral              |   8x7B, 8x22B    | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |
|             InternVL             |   v1.1 - v1.5    | MLLM |    Yes    |   Yes   |   Yes   |  Yes  |
|    InternVL2<sup>\[2\]</sup>     | 1 - 2B, 8B - 76B | MLLM |    Yes    |  Yes\*  |  Yes\*  |  Yes  |
| InternVL2.5(MPO)<sup>\[2\]</sup> |     1 - 78B      | MLLM |    Yes    |  Yes\*  |  Yes\*  |  Yes  |
|    InternVL3<sup>\[2\]</sup>     |     1 - 78B      | MLLM |    Yes    |  Yes\*  |  Yes\*  |  Yes  |
|           InternVL3.5            |   1 - 241BA28B   | MLLM |    Yes    |  Yes\*  |  Yes\*  |  No   |
|             ChemVLM              |     8B - 26B     | MLLM |    Yes    |   Yes   |   Yes   |  Yes  |
|          GLM-4.7-Flash           |       30B        | LLM  |    Yes    |   No    |   No    |  No   |
|             gpt-oss              |     20B,120B     | LLM  |    Yes    |   Yes   |   Yes   |  Yes  |

“-” 表示还没有验证。

```{note}
* [1] turbomind 引擎不支持 window attention。所以，对于应用了 window attention，并开启了对应的开关"use_sliding_window"的模型，比如 Qwen1.5 等，不支持推理
* [2] 当模型的 head_dim 非 128 时，turbomind 不支持它的 kv cache 4/8 bit 量化和推理。比如，llama3.2-1B，qwen2-0.5B，internvl2-1B 等等
```
