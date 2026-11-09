# Copyright (c) OpenMMLab. All rights reserved.
import enum
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from pydantic.dataclasses import dataclass as pydantic_dataclass

from .tokenizer import Tokenizer
from .utils import get_logger

logger = get_logger('lmdeploy')


class QuantPolicy(enum.IntEnum):
    """Quantization policy constants for KV cache."""
    NONE = 0
    INT4 = 4  # 4-bit KV cache
    INT8 = 8  # 8-bit KV cache


@dataclass
class GenerationConfig:
    """Generation parameters used by inference engines.

    Args:
        n: Define how many chat completion choices to generate for each
            input message. **Only 1** is supported now.
        max_new_tokens: The maximum number of tokens that can be
            generated in the chat completion
        do_sample:  Whether or not to use sampling, use greedy
            decoding otherwise. Default to be False.
        top_p: An alternative to sampling with temperature, called
            nucleus sampling, where the model considers the results of the
            tokens with top_p probability mass
        top_k: An alternative to sampling with temperature, where
            the model considers the top_k tokens with the highest probability
        min_p: Minimum token probability, which will be scaled by the
            probability of the most likely token. It must be a value between
            0 and 1. Typical values are in the 0.01-0.2 range, comparably
            selective as setting `top_p` in the 0.99-0.8 range (use the
            opposite of normal `top_p` values)
        temperature: Sampling temperature
        repetition_penalty: Penalty to prevent the model from
            generating repeated words or phrases. A value larger than
            1 discourages repetition
        ignore_eos: Indicator to ignore the eos_token_id or not
        random_seed: Seed used when sampling a token
        stop_words: Words that stop generating further tokens
        bad_words: Words that the engine will never generate
        stop_token_ids: list of tokens that stop the generation
            when they are generated. The returned output will not contain
            the stop tokens.
        bad_token_ids: list of tokens that the engine will never
            generate.
        min_new_tokens: The minimum numbers of tokens to generate,
            ignoring the number of tokens in the prompt.
        skip_special_tokens: Whether or not to remove special tokens
            in the decoding. Default to be True.
        spaces_between_special_tokens: Whether or not to add spaces
            around special tokens. The behavior of Fast tokenizers is to have
            this to False. This is setup to True in slow tokenizers.
        logprobs: Number of log probabilities to return per output token.
        response_format: Generate responses according to given formatting.
            Examples:

            .. code-block:: json

                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "test",
                        "schema": {
                        "properties": {
                            "name": {
                            "type": "string"
                            }
                        },
                        "required": ["name"],
                        "type": "object"
                        }
                    }
                }


            or,

            .. code-block:: json

                {
                    "type": "regex_schema",
                    "regex_schema": "call me [A-Za-z]{1,10}"
                }

            or, an XGrammar structural tag:

            .. code-block:: json

                {
                    "type": "structural_tag",
                    "format": {
                        "type": "tag",
                        "begin": "<answer>",
                        "content": {
                            "type": "regex",
                            "pattern": "[0-9]{1,3}"
                        },
                        "end": "</answer>"
                    }
                }
    """

    n: int = 1
    max_new_tokens: int = 512
    do_sample: bool = False
    top_p: float = 1.0
    top_k: int = 50
    min_p: float = 0.0
    temperature: float = 0.8
    repetition_penalty: float = 1.0
    ignore_eos: bool = False
    random_seed: int = None
    stop_words: list[str] = None
    bad_words: list[str] = None
    stop_token_ids: list[int] = None
    bad_token_ids: list[int] = None
    min_new_tokens: int = None
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    logprobs: int = None
    response_format: dict | None = None
    output_logits: Literal['all', 'generation'] = None
    output_last_hidden_state: Literal['all', 'generation'] = None
    include_stop_str_in_output: bool = False

    # return the perplexity (mean cross-entropy loss) of the input prompt,
    return_ppl: bool = False

    def convert_stop_bad_words_to_ids(self, tokenizer: Tokenizer):
        """Convert stop_words/bad_sords to ids and append the ids to
        stop_token_ids/bad_token_ids."""

        def special_word_token_ids(words):
            if words is not None:
                assert isinstance(words, list) and \
                    all(isinstance(elem, str) for elem in words), \
                    f'stop_words must be a list of str but got {type(words)}'
                indexes = []
                for word in words:
                    indexes += tokenizer.indexes_containing_token(word)
                return indexes
            return None

        stop_token_ids = special_word_token_ids(self.stop_words) or []
        bad_token_ids = special_word_token_ids(self.bad_words) or []
        stop_token_ids.extend(self.stop_token_ids or [])
        bad_token_ids.extend(self.bad_token_ids or [])
        self.stop_token_ids = list(set(stop_token_ids)) or None
        self.bad_token_ids = list(set(bad_token_ids)) or None

    def update_from_hf_gen_cfg(self, generation_config, tokenizer_eos_token_id):
        """Update the stop_token_ids."""
        stop_token_ids = set(self.stop_token_ids or [])

        # add tokenizer's eos_token_id
        if tokenizer_eos_token_id is not None:
            stop_token_ids.add(tokenizer_eos_token_id)

        # add eos_token_id from the model's generation config, if any.
        eos_token_id = generation_config.get('eos_token_id')
        if eos_token_id is not None:
            if isinstance(eos_token_id, int):
                stop_token_ids.add(eos_token_id)
            else:
                stop_token_ids.update(eos_token_id)

        self.stop_token_ids = list(stop_token_ids)

    def __post_init__(self):
        """Check input validation."""
        assert type(self.n) is int and self.n > 0, 'n is not a positive integer'
        assert self.top_p >= 0 and self.top_p <= 1  # [0, 1]
        assert self.top_k >= 0, 'top_k can not be a negative integer'
        assert self.temperature >= 0 and self.temperature <= 2  # [0,2]
        assert 0 <= self.min_p <= 1, \
            f'min_p should be in range [0, 1], but found {self.min_p}'


@pydantic_dataclass
class TurbomindEngineConfig:
    """TurboMind Engine config.

    Args:
        dtype: data type for model weights and activations. It can be
            one of the following values, ['auto', 'float16', 'bfloat16']
            The `auto` option will use FP16 precision for FP32 and FP16
            models, and BF16 precision for BF16 models.
        gemm_input_dtype: preferred GEMM input dtype. It can be None,
            'float16', 'bfloat16', or 'float8_e4m3'. This reorders eligible
            kernel families without changing the model dtype contract.
        model_format: the layout of the deployed model. It can be one
            of the following values [hf, awq, gptq, compressed-tensors,
            fp8, mxfp4]. `hf` means a Hugging Face model (.bin,
            .safetensors), `awq` and `gptq` mean grouped 4-bit
            weight-only checkpoints, `compressed-tensors` means
            pack-quantized grouped int4 checkpoints and is usually
            auto-detected from the input model config, `fp8` means
            blocked fp8 checkpoints, and `mxfp4` means MXFP4 expert
            weights. If it is not specified, i.e. None, it will be
            extracted from the input model
        tp: the number of GPU cards used in tensor parallelism,
            default to 1
        session_len: the max session length of a sequence, default to
            None
        max_batch_size: the max batch size during inference. If it is
            not specified, the engine will automatically set it according to
            the device
        cache_max_entry_count: the percentage of gpu memory occupied
            by the k/v cache.
            For versions of lmdeploy between `v0.2.0` and `v0.2.1`, it
            defaults to 0.5, depicting the percentage of TOTAL GPU memory to
            be allocated to the k/v cache.
            For lmdeploy versions greater than `v0.2.1`, it defaults to 0.8,
            signifying the percentage of FREE GPU memory to be reserved for
            the k/v cache.
            When it's an integer > 0, it represents the total number of k/v
            blocks.
        cache_chunk_size: The policy to apply for KV block from
            the block manager, default to -1.
        cache_block_seq_len: the length of the token sequence in
            a k/v block, default to 64
        enable_prefix_caching: enable cache prompts for block reuse,
            default to False
        cache_checkpoint_interval: minimum token gap between reusable
            recurrent-state checkpoints (CacheRegistry checkpoint_min_interval).
            Must be > 0. Default 4096.
        cache_prompt: partial prompt-boundary publication mode, one of
            'all' | 'auto'. 'all' publishes the reusable partial fork_to node at
            B = prompt_len - cache_prompt_boundary_skip whenever B is mid-block
            (and arms a recurrent-state checkpoint clamp when B is block-aligned),
            so a duplicate prompt skips prefill (costs one extra prefill forward +
            a partial block). 'auto' (default) does that only when the partial
            block holds image tokens (reusing vision-encoded KV) and is inert for
            text-only prompts. Requires enable_prefix_caching.
        cache_prompt_boundary_skip: number of trailing prompt tokens treated as
            the volatile generation-prompt suffix (e.g. a chat template's
            `<think>\n`) and excluded from the reusable prompt-boundary node, so
            the node ends at prompt_len - cache_prompt_boundary_skip. Default 1
            (exclude only the last token). Applies when cache_prompt is 'all' or
            'auto'.
        cache_generation: generated-block caching mode, one of
            'all' | 'auto' | 'none'. 'all' indexes full generated blocks and the
            terminal partial block, and adopts the terminal recurrent frontier
            checkpoint (exact multi-turn resume, costs a partial block). 'auto'
            (default) indexes full generated blocks only. 'none' indexes no
            generated blocks at all. Requires enable_prefix_caching.
        quant_policy: default to 0. For TurboMind, when k/v is quantized
            into int4 or int8, set it to 4 or 8, respectively
        rope_scaling_factor: scaling factor used for dynamic ntk,
            default to 0. TurboMind follows the implementation of transformer
            LlamaAttention
        use_logn_attn: whether or not to use log attn: default to False
        download_dir: Directory to download and load the weights,
            default to the default cache directory of huggingface.
        revision: The specific model version to use. It can be a branch
            name, a tag name, or a commit id. If unspecified, will use the
            default version.
        max_prefill_token_num: the number of tokens each iteration during
            prefill, default to 8192
        num_tokens_per_iter: the number of tokens processed in each
            forward pass. Working with `max_prefill_iters` enables the
            "Dynamic SplitFuse"-like scheduling
        max_prefill_iters: the max number of forward pass during prefill
            stage
        async_: enable async execution, default to 1 (enabled)
        devices: the used devices
        language_model_only: Whether to run as text-only LLM without loading
            vision/multimodal encoder modules.
        communicator: collective communicator, It can be one of the following values,
            ['nccl', 'cuda-ipc']. The `cuda-ipc` option only supports single-node
            multi-gpu communication.
        moe_a2a_backend: the backend of moe a2a communication, It can be one of the
            following values, ['auto', 'default', 'deepep'].
        hf_overrides: Huggingface overrides for the model.
            It can be used to override the default config of the model
        enable_metrics: enable metrics system
    """

    dtype: str = 'auto'
    gemm_input_dtype: str | None = None
    model_format: str | None = None
    tp: int = 1
    dp: int = 1
    cp: int = 1
    ep: int = 1
    device_num: int = None
    attn_tp_size: int = None
    attn_cp_size: int = None
    attn_dp_size: int = None
    mlp_tp_size: int = None
    mlp_dp_size: int = None
    outer_dp_size: int = None
    nnodes: int = 1
    node_rank: int = 0
    dist_init_addr: str | None = None
    devices: list[int] = None
    session_len: int | None = None
    max_batch_size: int = None
    cache_max_entry_count: float = 0.8
    cache_chunk_size: int = -1
    cache_block_seq_len: int = 64
    enable_prefix_caching: bool = False
    cache_checkpoint_interval: int = 4096
    cache_prompt: str = 'auto'
    cache_prompt_boundary_skip: int = 1
    cache_generation: str = 'auto'
    quant_policy: int = 0
    rope_scaling_factor: float = 0.0
    use_logn_attn: bool = False
    download_dir: str | None = None
    revision: str | None = None
    max_prefill_token_num: int = 8192
    num_tokens_per_iter: int = 0
    max_prefill_iters: int = 1
    async_: int = 1
    devices: list[int] | None = None
    language_model_only: bool = False
    communicator: str = 'nccl'
    moe_a2a_backend: str = 'auto'
    hf_overrides: dict[str, Any] | None = None
    enable_metrics: bool = True

    def __post_init__(self):
        """Check input validation."""
        assert self.dtype in ['auto', 'float16', 'bfloat16']
        assert self.gemm_input_dtype in (None, 'float16', 'bfloat16', 'float8_e4m3')
        assert self.tp >= 1, 'tp must be a positive integer'
        assert self.ep >= 1, 'ep must be a positive integer'
        assert self.cache_max_entry_count > 0, 'invalid cache_max_entry_count'
        try:
            self.quant_policy = QuantPolicy(self.quant_policy)
        except ValueError as e:
            raise ValueError(f'invalid quant_policy: {self.quant_policy}') from e
        assert self.rope_scaling_factor >= 0, 'invalid rope_scaling_factor'
        assert self.max_prefill_token_num >= 0, \
            'invalid max_prefill_token_num'
        assert self.num_tokens_per_iter >= 0, 'invalid num_tokens_per_iter'
        assert self.cache_prompt in ('all', 'auto'), 'invalid cache_prompt'
        assert self.cache_generation in ('all', 'auto', 'none'), 'invalid cache_generation'
        assert self.cache_checkpoint_interval > 0, 'invalid cache_checkpoint_interval'
        assert self.cache_prompt_boundary_skip >= 1, 'invalid cache_prompt_boundary_skip'
        assert self.async_ in (0, 1), 'async_ must be 0 (disabled) or 1 (enabled)'
        assert self.moe_a2a_backend in ('auto', 'default', 'deepep'), \
            'invalid moe_a2a_backend'


class ResponseType(enum.Enum):
    """Response type."""

    SUCCESS = enum.auto()
    FINISH = enum.auto()
    ENGINE_STOP_ERROR = enum.auto()
    SESSION_REPEAT = enum.auto()
    SESSION_NOT_EXIST = enum.auto()
    HANDLER_NOT_EXIST = enum.auto()
    INPUT_LENGTH_ERROR = enum.auto()
    INTERNAL_ENGINE_ERROR = enum.auto()
    CANCEL = enum.auto()
    PREFIX_CACHE_CONFLICT = enum.auto()
    NO_QUEUE = enum.auto()
    NOT_SUPPORTED = enum.auto()
    OUT_OF_MEMORY = enum.auto()


@dataclass
class Response:
    """Pack all response information together.

    Args:
        text: the response text from the server. If the output text is
            an empty str and the finish_reason is length, it means the session
            length is reached.
        generate_token_len: the response token length.
        input_token_len: the input prompt token length. Note that it may
            contains chat template part.
        finish_reason: the reason the model stopped
            generating tokens. This will be 'stop' if the model hit a natural
            stop point or a provided stop sequence, 'length' if the maximum
            number of tokens specified in the request was reached.
        token_ids: the output token ids.
        logprobs: the top logprobs for each output position.
        index: it refers to the position index of the input request batch.
    """
    text: str
    generate_token_len: int
    input_token_len: int
    finish_reason: Literal['stop', 'length', 'error', 'abort'] | None = None
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[dict[int, float]] = None
    logits: torch.Tensor = None
    last_hidden_state: torch.Tensor = None
    index: int = 0
    cached_tokens: int = 0
    error_code: str | None = None
    error_message: str | None = None

    def __str__(self):
        return f'text={self.text}\n{self._format_none_text_fields()}'

    def __repr__(self):
        return f'text={self.text!r}\n{self._format_none_text_fields()}'

    def _format_none_text_fields(self):
        fields = []
        fields.append(f'input_token_len={self.input_token_len}')
        fields.append(f'generate_token_len={self.generate_token_len}')
        fields.append(f'finish_reason="{self.finish_reason}"')
        fields.append(f'token_ids={self.token_ids}')
        fields.append(f'logprobs={self.logprobs}')

        # Helper function to format tensor information
        def _format_tensor(name: str, tensor: torch.Tensor | None) -> list[str]:
            if tensor is None:
                return [f'{name}=None']
            try:
                return [f'{name}.shape={tensor.shape}', f'{name}={tensor}']
            except:  # noqa
                # in case tensor is not torch.Tensor or has no shape
                return [f'{name}={tensor}']

        # Format tensor fields
        fields.extend(_format_tensor('logits', self.logits))
        fields.extend(_format_tensor('last_hidden_state', self.last_hidden_state))
        return '\n'.join(fields)

    def extend(self, other: 'Response') -> 'Response':
        """Extend this response with another response.

        This method merges the content of another Response into this one,
        similar to list.extend(). The text, token_ids, and logprobs are
        concatenated, while other fields are updated from the other response.

        Args:
            other: Another Response to append to this one.

        Returns:
            Self (for method chaining).
        """
        self.text += other.text
        self.generate_token_len = other.generate_token_len
        self.input_token_len = other.input_token_len
        self.finish_reason = other.finish_reason
        self.index = other.index
        if other.token_ids:
            self.token_ids += other.token_ids
        if other.logprobs is not None:
            self.logprobs = self.logprobs or []
            self.logprobs += other.logprobs
        self.error_code = other.error_code
        self.error_message = other.error_message
        return self


# modified from https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/__init__.py
class EventType(enum.IntEnum):
    """The type of request event.

    QUEUED - when the request was enqued by the engine
    SCHEDULED - when the request was first scheduled for execution
    PREEMPTED - the request has been put back in the waiting queue in order to make room for other requests to complete.
                It will be re-scheduled in future and re-start its prefill phase
    """
    QUEUED = 1
    SCHEDULED = 2
    PREEMPTED = 3  # FIXME, currently ignored for simplicity


# modified from https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/__init__.py
@dataclass
class EngineEvent:
    """A timestamped engine event associated with a request.

    Attributes:
        type: the type of an event associated with a request during its life cycle
        timestamp: the WALL-CLOCK time when the event happens.
    """
    type: EventType
    timestamp: float

    @classmethod
    def new_event(cls, event_type: EventType, timestamp: float | None = None) -> 'EngineEvent':
        # Timestamps MUST use wall-clock time (time.time()) to maintain consistency
        # between csrc(std::chrono::system_clock) and python
        timestamp = time.time() if timestamp is None else timestamp
        return cls(event_type, timestamp)


@dataclass
class ScheduleMetrics:
    active_seqs: int = 0
    waiting_seqs: int = 0
    cache_usage: float = 0.0
    prefix_cache_hit_rate: float = 0
    scheduler_tick: int = 0


@dataclass
class RequestMetrics:
    """Basic metrics for a request.

    Attributes:
        token_timestamp: A wall-clock time when a token is generated.
        engine_events: list of engine events during inference.
    """
    token_timestamp: float = 0.0
    engine_events: list[EngineEvent] = field(default_factory=list)
    spec_info: dict[str, Any] | None = None
    cached_tokens: int = 0


@dataclass
class EngineOutput:
    """Engine output from the TurboMind engine.

    Args:
        status: the response type.
        token_ids: the newly generated token ids in each iteration.
        logprobs: the top logprobs for each output position.
        req_metrics: request metrics information
        ce_loss: the summed, unnormalized cross-entropy (NLL) of the input
            prompt, available when ``GenerationConfig.return_ppl`` is set.
    """
    status: ResponseType
    token_ids: list[int]
    logprobs: list[dict[int, float]] = None
    logits: torch.Tensor = None
    last_hidden_state: torch.Tensor = None
    req_metrics: RequestMetrics | None = None
    ce_loss: float = None


SPECULATIVE_METHODS = ('mtp', 'eagle3')


@dataclass
class SpeculativeConfig:
    """Speculative decoding config.

    Args:
        method: the speculative decoding method, one of ``SPECULATIVE_METHODS``.
            ``mtp`` uses the target checkpoint's multi-token-prediction layers;
            ``eagle3`` requires a draft model path.
        model: the path of speculative model.
        num_speculative_tokens: number of generated token of draft model per step
    """
    method: str
    model: str = ''
    num_speculative_tokens: int = 1

    def __post_init__(self):
        if self.method not in SPECULATIVE_METHODS:
            raise ValueError(f'unsupported speculative method {self.method!r}; '
                             f'supported methods are {list(SPECULATIVE_METHODS)}')
