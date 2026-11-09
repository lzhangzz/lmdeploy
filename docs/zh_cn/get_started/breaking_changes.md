# 不兼容变更：LMDeploy 仅保留 TurboMind 引擎

LMDeploy 现在只提供一个推理引擎 [TurboMind](../inference/turbomind.md)，并且仅支持 NVIDIA CUDA GPU。PyTorch 引擎（`lmdeploy.pytorch`）以及所有依赖它的功能均已移除。此次变更不提供兼容层：使用已移除接口的代码会直接报导入错误或参数错误，而不会静默回退。

本页列出被移除的内容，以及在有替代方案时应使用的替代方案。

## 安装与平台

- 仅支持 NVIDIA CUDA 构建。昇腾、沐曦（MACA）、寒武纪和 AMD ROCm 的安装方式、对应的依赖文件（`requirements_{ascend,camb,maca,rocm}.txt`）以及昇腾 Docker 镜像均已移除。
- 不再识别 `LMDEPLOY_TARGET_DEVICE` 和 `DISABLE_TURBOMIND`。从源码构建时总会编译 TurboMind 原生扩展。
- 不再依赖 `cloudpickle`、`pyzmq` 和 `ray`。
- Docker 镜像不再安装 PyTorch 引擎所需的算子和传输库：DeepEP（Python）、DeepGEMM、FlashMLA、fast-hadamard-transform、FlashAttention 3、NVSHMEM、GDRCopy、DLSlime 和 tile-kernels。TurboMind 原生的 DeepEP 集成从源码构建，不需要这些包。

## 引擎选择

- 移除了 `PytorchEngineConfig`，请使用 `TurbomindEngineConfig`。
- `--backend pytorch` 会被拒绝。`--backend turbomind` 仍然可用，并且是唯一合法的取值。
- TurboMind 不支持的模型架构现在会直接报错，而不会回退到 PyTorch 引擎。参见[支持的模型](../supported_models/supported_models.md)。

## 不再支持的模型

以下模型此前只能通过 PyTorch 引擎运行，或已不在 TurboMind 的架构注册表中：

- LLM：Llama4、Mistral、ChatGLM2、GLM-4、GLM-4-0414、GLM-4.5、GLM-4.5-Air、GLM-5、GLM-5.2、CodeGeeX4、DeepSeek-MoE、DeepSeek-V2、DeepSeek-V2.5、DeepSeek-V3、DeepSeek-V3.2、DeepSeek-V4、Hy3、Qwen3-Next、MiniCPM3、Gemma、Gemma2、Phi-3-mini、Phi-3.5-mini、Phi-3.5-MoE、Phi-4-mini、SDAR
- VLM：LLaVA（1.5、1.6）及 llava-hf 模型、Qwen3-VL、Qwen3-Omni、Kimi-K2.6、DeepSeek-VL、DeepSeek-VL2、MiniCPM-Llama3-V-2_5、MiniCPM-V-2_6、Phi-3-vision、Phi-3.5-vision、CogVLM、CogVLM2、GLM-4V、GLM-4.1V-Thinking、Molmo、Gemma3、Intern-S1-Pro、Mono-InternVL、InternVL2-4B（基于 Phi-3）
- 奖励模型
- 音频输入（Qwen3-Omni）和时序数据输入（Intern-S1-Pro）

上述 VLM 的 Python 视觉适配代码也已删除，因此 `lmdeploy lite` 无法再加载或量化这些模型。Qwen3-VL 的预处理器仍然保留，因为 Qwen3.5 的适配代码基于它实现。

## 移除的功能

- LoRA 适配器（`adapters`、`--adapters`、`adapter_name`）。
- W8A8 / SmoothQuant：`lmdeploy lite smooth_quant` 命令、对应的 Python 接口以及 W8A8 模型。
- PD 分离部署（DistServe）：`--role`、`--migration-backend`、`--kv-transfer-config`、代理服务的 `--serving-strategy` 选项，以及 `/distserve/*` 接口。
- 基于 Ray 的分布式执行（`--distributed-executor-backend`）。
- 扩散语言模型解码（`--dllm-*`）。
- Logits 处理器（`GenerationConfig.logits_processors`）和 `logit_bias`，后者现在会被 OpenAI 接口拒绝。
- 输入 token 的 logprob（`logprob_start_len`、`input_token_logprobs`、`input_top_logprobs`）以及 `--logprobs-mode`。
- 路由专家输出（`return_routed_experts`、`--enable-return-routed-experts`）。
- 重复 n-gram 惩罚（`repetition_ngram_size`、`repetition_ngram_threshold`）。
- int4 和 int8 以外的 KV cache 量化策略：`quant_policy` 只接受 `0`、`4` 和 `8`。FP8 KV cache（`16`、`17`）和 TurboQuant（`42`）已移除。
- 投机解码方法 `deepseek_mtp`、`qwen3_5_mtp` 和 `dflash`（`--speculative-dflash-block-size`）。TurboMind 支持 `mtp`（Qwen3.5 系列）和 `eagle3`，参见[投机解码](../advance/spec_decoding.md)。
- 投机解码与结构化输出的组合。TurboMind 不会对投机 token 应用语法掩码，因此启用投机解码时 `response_format` 不会生效。
- 在线权重更新：`/update_weights`、`/update_weights_from_ipc`、`/init_weights_update_group`、`/update_weights_from_distributed`、`/destroy_weights_update_group`、`TurbomindEngineConfig.empty_init` 和 `lmdeploy.utils.serialize_state_dict`。
- 引擎休眠与唤醒：`/sleep`、`/wakeup` 和 `/is_sleeping`。
- `/pooling` 接口。
- 音频和时序数据消息内容（`audio`、`audio_url`、`time_series`、`time_series_url`）。包含这些内容的请求会被拒绝。`lmdeploy.vl.load_audio`、`load_time_series`、`encode_audio_base64` 和 `encode_time_series_base64` 已移除。
- TurboMind 无法运行的模型所对应的对话模板：`cogvlm`、`llava-v1`、`llava-chatml`、`chatglm`、`deepseek-vl`、`deepseek-vl2`、`deepseek-v4`、`deepseek-v32` 和 `deepseek-v3.2`。`mistral` 模板名称已移除，`mixtral` 仍然保留。
- reasoning parser `deepseek-v3`、`deepseek-v32`、`deepseek-v3.2` 和 `deepseek-v4`，以及 tool parser `deepseek-v32`、`deepseek-v4`、`kimi_k2` 和 `kimi-k2`。
- `lmdeploy lite` 对 Phi-3、ChatGLM、Mistral 和 Mini-Gemini 模型的支持。
- `VisionConfig`、`vision_config` 参数和 `--vision-max-batch-size`。TurboMind 原生运行视觉编码器，传入 `vision_config` 现在会抛出 `TypeError`。
- `lmdeploy.tokenizer` 中的 `ChatGLMTokenizer` 和 `ChatGLM4Tokenizer`。
- 仅 PyTorch 引擎使用的选项，例如 `--eager-mode`、`--block-size`、`--num-gpu-blocks`、`--cudagraph-capture-batch-sizes`、`--enable-microbatch`、`--enable-eplb`、`--quant-dtype` 和 `--device`。

## 移除的请求字段

以下 `GenerationConfig` 字段已移除，传入时会抛出 `TypeError`：`logits_processors`、`with_cache`、`preserve_cache`、`migration_request`、`return_routed_experts`、`logprob_start_len`、`repetition_ngram_size` 和 `repetition_ngram_threshold`。

OpenAI 兼容接口和 `/generate` 的请求结构中也移除了对应字段。未知字段会被忽略，但 `logit_bias` 会被拒绝。Anthropic 接口会拒绝 `return_logprob`。

## 其他行为变化

- `benchmark/profile_throughput.py` 现在会把 `--speculative-algorithm` 等投机解码参数传给 TurboMind，此前这些参数在 TurboMind 下会被忽略。
