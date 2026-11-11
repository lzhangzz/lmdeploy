# Offline Inference Pipeline

In this tutorial, We will present a list of examples to introduce the usage of `lmdeploy.pipeline`.

You can overview the detailed pipeline API in [this](https://lmdeploy.readthedocs.io/en/latest/api/pipeline.html) guide.

## Usage

### A 'Hello, world' example

```python
from lmdeploy import pipeline

pipe = pipeline('internlm/internlm2_5-7b-chat')
response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
print(response)
```

In this example, the pipeline by default allocates a predetermined percentage of GPU memory for storing k/v cache. The ratio is dictated by the parameter `TurbomindEngineConfig.cache_max_entry_count`: it is the proportion of **free** GPU memory reserved for the k/v cache after the model weights are loaded, defaulting to 0.8. The k/v cache memory is allocated once and reused, which is why a built pipeline consumes a substantial amount of GPU memory. If you encounter an Out Of Memory (OOM) error, decrease the ratio:

```python
from lmdeploy import pipeline, TurbomindEngineConfig

# decrease the ratio of the k/v cache occupation to 20%
backend_config = TurbomindEngineConfig(cache_max_entry_count=0.2)

pipe = pipeline('internlm/internlm2_5-7b-chat',
                backend_config=backend_config)
response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
print(response)
```

See [Engine configuration](#engine-configuration) for the rest of the cache knobs.

### Set tensor parallelism

```python
from lmdeploy import pipeline, TurbomindEngineConfig

backend_config = TurbomindEngineConfig(tp=2)
pipe = pipeline('internlm/internlm2_5-7b-chat',
                backend_config=backend_config)
response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
print(response)
```

### Set sampling parameters

```python
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

backend_config = TurbomindEngineConfig(tp=2)
gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
pipe = pipeline('internlm/internlm2_5-7b-chat',
                backend_config=backend_config)
response = pipe(['Hi, pls intro yourself', 'Shanghai is'],
                gen_config=gen_config)
print(response)
```

### Apply OpenAI format prompt

```python
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

backend_config = TurbomindEngineConfig(tp=2)
gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
pipe = pipeline('internlm/internlm2_5-7b-chat',
                backend_config=backend_config)
prompts = [[{
    'role': 'user',
    'content': 'Hi, pls intro yourself'
}], [{
    'role': 'user',
    'content': 'Shanghai is'
}]]
response = pipe(prompts,
                gen_config=gen_config)
print(response)
```

### Apply streaming output

```python
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

backend_config = TurbomindEngineConfig(tp=2)
gen_config = GenerationConfig(top_p=0.8,
                              top_k=40,
                              temperature=0.8,
                              max_new_tokens=1024)
pipe = pipeline('internlm/internlm2_5-7b-chat',
                backend_config=backend_config)
prompts = [[{
    'role': 'user',
    'content': 'Hi, pls intro yourself'
}], [{
    'role': 'user',
    'content': 'Shanghai is'
}]]
for item in pipe.stream_infer(prompts, gen_config=gen_config):
    print(item)
```

### Get logits for generated tokens

```python
from lmdeploy import pipeline, GenerationConfig

pipe = pipeline('internlm/internlm2_5-7b-chat')

gen_config=GenerationConfig(output_logits='generation',
                            max_new_tokens=10)
response = pipe(['Hi, pls intro yourself', 'Shanghai is'],
                gen_config=gen_config)
logits = [x.logits for x in response]
```

### Get last layer's hidden states for generated tokens

```python
from lmdeploy import pipeline, GenerationConfig

pipe = pipeline('internlm/internlm2_5-7b-chat')

gen_config=GenerationConfig(output_last_hidden_state='generation',
                            max_new_tokens=10)
response = pipe(['Hi, pls intro yourself', 'Shanghai is'],
                gen_config=gen_config)
hidden_states = [x.last_hidden_state for x in response]
```

### Calculate ppl

```python
from transformers import AutoTokenizer
from lmdeploy import pipeline

model_repoid_or_path = 'internlm/internlm2_5-7b-chat'
pipe = pipeline(model_repoid_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_repoid_or_path, trust_remote_code=True)
messages = [
   {"role": "user", "content": "Hello, how are you?"},
]
input_ids = tokenizer.apply_chat_template(messages)

# ppl is a list of float numbers
ppl = pipe.get_ppl(input_ids)
print(ppl)
```

```{note}
- When input_ids is too long, an OOM (Out Of Memory) error may occur. Please apply it with caution
- get_ppl returns the cross entropy loss without applying the exponential operation afterwards
```

### Release pipeline

You can release the pipeline explicitly by calling its `close()` method, or alternatively, use the `with` statement as demonstrated below:

```python
from lmdeploy import pipeline

with pipeline('internlm/internlm2_5-7b-chat') as pipe:
    response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
    print(response)
```

## Engine configuration

Every knob below is a field of `TurbomindEngineConfig`. The list is not
exhaustive; the remaining fields are documented in the class docstring under
the [pipeline API](../api/pipeline.rst).

### k/v cache size

k/v cache memory is determined by `cache_block_seq_len` and
`cache_max_entry_count`. TurboMind implements Paged Attention, so the cache is
managed in blocks.

`cache_block_seq_len` is the number of token positions in one k/v block,
defaulting to 64. The memory of a block follows from the model geometry:

```
cache_block_seq_len * num_layer * kv_head_num * size_per_head * 2 * sizeof(kv_data_type)
```

For llama2-7b storing k/v as `half`, one block is
`64 * 32 * 32 * 128 * 2 * sizeof(half) = 32MB`.

With context parallelism the formula still describes the physical per-rank
block; the scheduler's block spans `cache_block_seq_len * cp` global tokens,
because those tokens are interleaved across the CP ranks.

`cache_max_entry_count` takes two forms:

- A fraction in (0, 1) is the proportion of **free** GPU memory reserved for
  k/v blocks; the default is 0.8. On an A100-80G running a 13B model, which
  leaves 54G free, the blocks get `(80 - 26) * 0.8 = 43.2G`.
- An integer greater than 0 is an absolute block count.

`cache_chunk_size` controls how many blocks are allocated at a time when more
are needed:

- An integer greater than 0 allocates that many blocks.
- `-1` allocates `cache_max_entry_count` blocks.
- `0` allocates `sqrt(cache_max_entry_count)` blocks.

### Prefix caching

`enable_prefix_caching` turns on reuse of the k/v blocks of a prompt prefix
that several requests share, typically a system prompt. The longer the shared
prefix, the larger the saving. The default is `False`.

Because a full k/v block is the smallest unit of reuse, the threshold for a
prefix hit is `cache_block_seq_len` tokens, or `cache_block_seq_len * cp` with
context parallelism.

### Partial-block boundary reuse

Two mode knobs control whether partial-block prefix nodes are published at the
prompt and generation boundaries: `cache_prompt` and `cache_generation`. Both
require `enable_prefix_caching`. A published node carries the partial block's
k/v and, on a recurrent/hybrid model (e.g. one with GatedDeltaNet layers), a
recurrent-state checkpoint. The node costs extra VRAM and copy bandwidth, so
prefer the default `'auto'` unless the reuse clearly pays off.

```python
from lmdeploy import pipeline, TurbomindEngineConfig

backend_config = TurbomindEngineConfig(
    enable_prefix_caching=True,
    cache_prompt='all',
    cache_generation='all',
)
pipe = pipeline('your-model', backend_config=backend_config)
```

- `cache_prompt`: partial prompt-boundary publication mode, `'all'` or
  `'auto'` (default `'auto'`). `'all'` publishes a reusable prompt-boundary
  node at `B = prompt_len - cache_prompt_boundary_skip` whenever `B` is
  mid-block (and arms a recurrent-state checkpoint clamp when `B` is
  block-aligned), so a duplicate prompt skips prefill. `'auto'` does that only
  when the partial block holds image tokens (reusing vision-encoded KV) and is
  inert for text-only prompts. Costs one extra prefill forward for the
  producing request and (when `B` is mid-block) one partial cache block.
- `cache_prompt_boundary_skip`: number of trailing prompt tokens treated as
  the volatile generation-prompt suffix (e.g. a chat template's `<think>\n`)
  and excluded from the reusable prompt-boundary node, moving it to
  `prompt_len - cache_prompt_boundary_skip`. Applies when `cache_prompt` is
  `'all'` or `'auto'`. Default 1 (exclude only the last token). Increase it
  for thinking models whose chat template appends a multi-token suffix that
  the next turn drops from history.
- `cache_generation`: generated-block caching mode, `'all'`, `'auto'`
  (default), or `'none'`. `'all'` indexes full generated blocks and the
  terminal partial block, and adopts the terminal recurrent frontier
  checkpoint (exact multi-turn resume). `'auto'` indexes full generated blocks
  only. `'none'` indexes no generated blocks. Costs one partial cache block
  when `'all'` indexes a terminal partial block. Full-block checkpoints at block
  boundaries are always published regardless of this setting.

### KV cache quantization

`quant_policy=4` stores the k/v cache as int4, `quant_policy=8` as int8; the
default `0` stores it in the model dtype. See
[KV cache quantization](../quantization/kv_quant.md).

### Long context

`rope_scaling_factor` (default `0`, disabled) enables the Dynamic NTK option
of RoPE, which lets the model handle text longer than its training window. Set
it to a value greater than 1 together with a matching `session_len`. See
[Context length extrapolation](../advance/long_context.md).

`use_logn_attn` turns on LogN attention scaling; the default is `False`.

### Recurrent state dtype

For models with GatedDeltaNet layers, TurboMind stores the recurrent state in
the model activation dtype by default. Set `state_dtype='float32'` in
`TurbomindEngineConfig` to store it in FP32 instead; the convolution state
stays in the activation dtype. FP32 improves numerical precision at twice the
recurrent-state memory.

```python
from lmdeploy import pipeline, TurbomindEngineConfig

pipe = pipeline('<model-path>', backend_config=TurbomindEngineConfig(state_dtype='float32'))
```

When `state_dtype` is left unset, `LMDEPLOY_FP32_MAMBA_SSM_DTYPE=1` selects FP32
as well, which also works with the CLI:

```bash
LMDEPLOY_FP32_MAMBA_SSM_DTYPE=1 lmdeploy serve api_server <model-path>
```

## FAQs

- **RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase**.

  If you got this error, please make sure the python script has the following entry point

  ```python
  if __name__ == '__main__':
  ```

  Generally, in the context of multi-threading or multi-processing, it might be necessary to ensure that initialization code is executed only once. In this case, `if __name__ == '__main__':` can help to ensure that these initialization codes are run only in the main program, and not repeated in each newly created process or thread.

- To customize a chat template, please refer to [chat_template.md](../advance/chat_template.md).
