# Breaking Changes: TurboMind-only LMDeploy

LMDeploy now ships a single inference engine, [TurboMind](../inference/turbomind.md), and supports NVIDIA CUDA GPUs only. The PyTorch engine (`lmdeploy.pytorch`) and everything that depended on it have been removed. There is no compatibility shim: code that uses a removed API fails with an import or argument error instead of falling back silently.

This page lists what was removed and what to use instead, where a replacement exists.

## Installation and platforms

- Only NVIDIA CUDA builds are supported. Ascend, MACA (MetaX), Cambricon and AMD ROCm installations, their requirement files (`requirements_{ascend,camb,maca,rocm}.txt`) and the Ascend Docker images were removed.
- `LMDEPLOY_TARGET_DEVICE` and `DISABLE_TURBOMIND` are no longer recognized. A source build always compiles the TurboMind native extension.
- `cloudpickle`, `pyzmq` and `ray` are no longer dependencies.
- The Docker image no longer installs the PyTorch-engine kernels and transports: DeepEP (Python), DeepGEMM, FlashMLA, fast-hadamard-transform, FlashAttention 3, NVSHMEM, GDRCopy, DLSlime and tile-kernels. TurboMind's native DeepEP integration is built from source and does not need them.

## Engine selection

- `PytorchEngineConfig` was removed. Use `TurbomindEngineConfig`.
- `--backend pytorch` is rejected. `--backend turbomind` is still accepted and is the only valid value.
- Models whose architecture TurboMind does not support now fail with an explicit error instead of falling back to the PyTorch engine. See [supported models](../supported_models/supported_models.md).

## Models no longer supported

The following models were only available through the PyTorch engine, or are no longer in the TurboMind architecture registry:

- LLMs: Llama4, Mistral, ChatGLM2, GLM-4, GLM-4-0414, GLM-4.5, GLM-4.5-Air, GLM-5, GLM-5.2, CodeGeeX4, DeepSeek-MoE, DeepSeek-V2, DeepSeek-V2.5, DeepSeek-V3, DeepSeek-V3.2, DeepSeek-V4, Hy3, Qwen3-Next, MiniCPM3, Gemma, Gemma2, Phi-3-mini, Phi-3.5-mini, Phi-3.5-MoE, Phi-4-mini, SDAR
- VLMs: LLaVA (1.5, 1.6) and llava-hf models, Qwen3-VL, Qwen3-Omni, Kimi-K2.6, DeepSeek-VL, DeepSeek-VL2, MiniCPM-Llama3-V-2_5, MiniCPM-V-2_6, Phi-3-vision, Phi-3.5-vision, CogVLM, CogVLM2, GLM-4V, GLM-4.1V-Thinking, Molmo, Gemma3, Intern-S1-Pro, Mono-InternVL, InternVL2-4B (Phi-3 based)
- Reward models
- Audio input (Qwen3-Omni) and time-series input (Intern-S1-Pro)

The Python vision adapters for these VLMs were removed as well, so `lmdeploy lite` can no longer load or quantize them. The Qwen3-VL preprocessor is kept because the Qwen3.5 adapter builds on it.

## Features removed

- LoRA adapters (`adapters`, `--adapters`, `adapter_name`).
- W8A8 / SmoothQuant: the `lmdeploy lite smooth_quant` command, its Python API and W8A8 checkpoints.
- PD disaggregation (DistServe): `--role`, `--migration-backend`, `--kv-transfer-config` and the proxy `--serving-strategy` option, plus the `/distserve/*` endpoints.
- Ray-based distributed execution (`--distributed-executor-backend`).
- Diffusion LLM decoding (`--dllm-*`).
- Logits processors (`GenerationConfig.logits_processors`) and `logit_bias`, which is now rejected by the OpenAI endpoint.
- Input-token logprobs (`logprob_start_len`, `input_token_logprobs`, `input_top_logprobs`) and `--logprobs-mode`.
- Routed-experts output (`return_routed_experts`, `--enable-return-routed-experts`).
- Repetition n-gram penalty (`repetition_ngram_size`, `repetition_ngram_threshold`).
- KV-cache quantization policies other than int4 and int8: `quant_policy` accepts only `0`, `4` and `8`. FP8 KV cache (`16`, `17`) and TurboQuant (`42`) were removed.
- Speculative decoding methods `deepseek_mtp`, `qwen3_5_mtp` and `dflash` (`--speculative-dflash-block-size`). TurboMind supports `mtp` (Qwen3.5 family) and `eagle3`; see [speculative decoding](../advance/spec_decoding.md).
- Structured output combined with speculative decoding. TurboMind does not apply grammar masks to speculative tokens, so `response_format` has no effect while speculative decoding is enabled.
- Online weight updates over IPC, NCCL groups or checkpoint-engine: `/update_weights_from_ipc`, `/init_weights_update_group`, `/update_weights_from_distributed` and `/destroy_weights_update_group`. `/update_weights` remains, without `load_format='flattened_bucket'`.
- The `/pooling` endpoint.
- Audio and time-series message content (`audio`, `audio_url`, `time_series`, `time_series_url`). Requests that contain them are rejected. `lmdeploy.vl.load_audio`, `load_time_series`, `encode_audio_base64` and `encode_time_series_base64` were removed.
- Chat templates for models TurboMind cannot run: `cogvlm`, `llava-v1`, `llava-chatml`, `chatglm`, `deepseek-vl`, `deepseek-vl2`, `deepseek-v4`, `deepseek-v32` and `deepseek-v3.2`. The `mistral` template name was removed; `mixtral` remains.
- Reasoning parsers `deepseek-v3`, `deepseek-v32`, `deepseek-v3.2` and `deepseek-v4`, and tool parsers `deepseek-v32`, `deepseek-v4`, `kimi_k2` and `kimi-k2`.
- `lmdeploy lite` support for Phi-3, ChatGLM, Mistral and Mini-Gemini checkpoints.
- `VisionConfig`, the `vision_config` argument and `--vision-max-batch-size`. TurboMind runs vision encoders natively, so passing `vision_config` now raises a `TypeError`.
- `ChatGLMTokenizer` and `ChatGLM4Tokenizer` in `lmdeploy.tokenizer`.
- PyTorch-engine-only options such as `--eager-mode`, `--block-size`, `--num-gpu-blocks`, `--cudagraph-capture-batch-sizes`, `--enable-microbatch`, `--enable-eplb`, `--quant-dtype` and `--device`.

## Removed request fields

The following `GenerationConfig` fields were removed and now raise `TypeError`: `logits_processors`, `with_cache`, `preserve_cache`, `migration_request`, `return_routed_experts`, `logprob_start_len`, `repetition_ngram_size` and `repetition_ngram_threshold`.

The corresponding HTTP request fields were removed from the OpenAI-compatible and `/generate` schemas. Unknown fields are ignored, except `logit_bias`, which is rejected. The Anthropic endpoint rejects `return_logprob`.

## Other behavior changes

- `benchmark/profile_throughput.py` now passes `--speculative-algorithm` and related options to TurboMind. They were previously ignored for TurboMind.
- `lmdeploy.utils.serialize_state_dict` serializes only `name -> tensor` mappings; the flattened-bucket format was removed together with `FlattenedTensorBucket`.
