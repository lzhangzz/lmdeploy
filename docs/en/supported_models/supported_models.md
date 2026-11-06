# Supported Models

The following table details the models supported by LMDeploy's TurboMind engine on NVIDIA CUDA GPUs. Models whose architecture is not listed here cannot be served by LMDeploy.

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

"-" means not verified yet.

```{note}
* [1] The TurboMind engine doesn't support window attention. Therefore, models that have applied window attention and have the corresponding switch "use_sliding_window" enabled, such as Qwen1.5, are not supported.
* [2] When the head_dim of a model is not 128, such as llama3.2-1B, qwen2-0.5B and internvl2-1B, turbomind doesn't support its kv cache 4/8 bit quantization and inference
```
