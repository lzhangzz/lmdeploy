# Copyright (c) OpenMMLab. All rights reserved.

import argparse
import json
import re
import sys
from collections import defaultdict
from typing import Any


class DefaultsAndTypesHelpFormatter(argparse.HelpFormatter):
    """Formatter to output default value and type in help information."""

    def _get_help_string(self, action):
        """Add default and type info into help."""
        help = action.help
        if '%(default)' not in action.help:
            if action.default is not argparse.SUPPRESS:
                defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
                if (action.option_strings or action.nargs in defaulting_nargs) and 'default' not in help.lower():
                    if not help.endswith('.'):
                        help += '.'
                    help += ' Default: %(default)s'
                if action.type:
                    if not help.endswith('.'):
                        help += '.'
                    help += ' Type: %(type)s'
        return help


def convert_args(args):
    """Convert args to dict format."""
    special_names = ['run', 'command']
    kwargs = {k[0]: k[1] for k in args._get_kwargs() if k[0] not in special_names}
    return kwargs


def get_chat_template(chat_template: str | None, model_path: str | None = None):
    """Get chat template config.

    Args:
        chat_template(str | None): it could be a builtin chat template name, or a chat template json file
        model_path(str | None): the model path, passed through to the chat template config
    """
    import os

    from lmdeploy.model import ChatTemplateConfig
    if chat_template:
        if os.path.isfile(chat_template):
            return ChatTemplateConfig.from_json(chat_template)
        else:
            from lmdeploy.model import MODELS
            assert chat_template in MODELS.module_dict.keys(), \
                f"chat template '{chat_template}' is not " \
                f'registered. The builtin chat templates are: ' \
                f'{MODELS.module_dict.keys()}'
            return ChatTemplateConfig(model_name=chat_template, model_path=model_path)
    else:
        return None


def get_speculative_config(args):
    """Get speculative config from args."""
    from lmdeploy.messages import SpeculativeConfig
    speculative_config = None
    if args.speculative_algorithm is not None:
        speculative_config = SpeculativeConfig(
            method=args.speculative_algorithm,
            model=args.speculative_draft_model,
            num_speculative_tokens=args.speculative_num_draft_tokens,
        )
    return speculative_config


class ArgumentHelper:
    """Helper class to add unified argument."""

    @staticmethod
    def model_name(parser):
        """Add argument model_name to parser."""

        return parser.add_argument('--model-name',
                                   type=str,
                                   default=None,
                                   help='The name of the served model. It can be accessed '
                                   'by the RESTful API `/v1/models`. If it is not specified, '
                                   '`model_path` will be adopted')

    @staticmethod
    def dtype(parser, default: str = 'auto'):
        return parser.add_argument('--dtype',
                                   type=str,
                                   default=default,
                                   choices=['auto', 'float16', 'bfloat16'],
                                   help='data type for model weights and activations. '
                                   'The "auto" option will use FP16 precision '
                                   'for FP32 and FP16 models, and BF16 precision '
                                   'for BF16 models. This option will be ignored if '
                                   'the model is a quantized model')

    @staticmethod
    def model_format(parser, default: str = None):
        return parser.add_argument('--model-format',
                                   type=str,
                                   default=default,
                                   choices=['hf', 'awq', 'gptq', 'compressed-tensors', 'fp8', 'mxfp4'],
                                   help='The format of input model. `hf` means `hf_llama`, '
                                   '`awq` and `gptq` refer to 4-bit grouped quantization, '
                                   '`compressed-tensors` refers to pack-quantized grouped int4 checkpoints and is '
                                   'usually auto-detected from the model config, `fp8` refers to blocked fp8 '
                                   'checkpoints, and `mxfp4` refers to MXFP4 expert weights.')

    @staticmethod
    def revision(parser, default: str = None):
        return parser.add_argument('--revision',
                                   type=str,
                                   default=default,
                                   help='The specific model version to use. '
                                   'It can be a branch name, a tag name, or a commit id. '
                                   'If unspecified, will use the default version.')

    @staticmethod
    def download_dir(parser, default: str = None):
        return parser.add_argument('--download-dir',
                                   type=str,
                                   default=default,
                                   help='Directory to download and load the weights, '
                                   'default to the default cache directory of huggingface.')

    @staticmethod
    def tp(parser):
        """Add argument tp to parser."""

        return parser.add_argument('--tp',
                                   type=int,
                                   default=1,
                                   help='GPU number used in tensor parallelism. Should be 2^n')

    @staticmethod
    def dp(parser):
        """Add argument dp to parser."""

        return parser.add_argument('--dp',
                                   type=int,
                                   default=1,
                                   help='data parallelism')

    @staticmethod
    def ep(parser):
        """Add argument ep to parser."""

        return parser.add_argument('--ep',
                                   type=int,
                                   default=1,
                                   help='expert parallelism')

    @staticmethod
    def cp(parser):
        """Add argument cp to parser."""

        return parser.add_argument(
            '--cp',
            type=int,
            default=1,
            help='context parallelism size in attention, tp must be a multiple of cp.')

    @staticmethod
    def node_rank(parser):
        """Add argument node_rank to parser."""

        return parser.add_argument('--node-rank', type=int, default=0, help='The current node rank.')

    @staticmethod
    def num_nodes(parser):
        """Add argument num_nodes to parser."""

        return parser.add_argument('--nnodes', type=int, default=1, help='The total node nums')

    @staticmethod
    def dist_init_addr(parser):
        """Add argument dist_init_addr to parser."""

        return parser.add_argument('--dist-init-addr', type=str, default=None)

    @staticmethod
    def session_id(parser):
        """Add argument session_id to parser."""

        return parser.add_argument('--session-id', type=int, default=1, help='The identical id of a session')

    @staticmethod
    def session_len(parser, default: int = None):
        return parser.add_argument('--session-len',
                                   type=int,
                                   default=default,
                                   help='The max session length of a sequence')

    @staticmethod
    def max_batch_size(parser):
        """Add argument max_batch_size to parser."""

        return parser.add_argument('--max-batch-size',
                                   type=int,
                                   default=None,
                                   help='Maximum batch size. If not specified, the engine will '
                                   'automatically set it according to the device')

    @staticmethod
    def quant_policy(parser, default: int = 0):
        """Add argument quant_policy to parser."""

        from lmdeploy.messages import QuantPolicy

        supported = list(QuantPolicy)
        _aliases = {p.name.lower(): p.value for p in supported}

        def _parse(x):
            key = x.lower()
            if key in _aliases:
                return _aliases[key]
            try:
                v = int(x)
            except ValueError:
                v = None
            if v not in supported:
                raise argparse.ArgumentTypeError(f'invalid quant_policy {x!r}; choose from none/int4/int8 (or 0/4/8)')
            return v

        return parser.add_argument('--quant-policy',
                                   type=_parse,
                                   default=default,
                                   help='KV cache quant policy: none/int4/int8 (or 0/4/8).')

    @staticmethod
    def rope_scaling_factor(parser):
        """Add argument rope_scaling_factor to parser."""

        return parser.add_argument('--rope-scaling-factor', type=float, default=0.0, help='Rope scaling factor')

    @staticmethod
    def hf_overrides(parser):
        """Add argument hf_overrides to parser."""
        return parser.add_argument('--hf-overrides',
                                   type=json.loads,
                                   default=None,
                                   help='Extra arguments to be forwarded to the HuggingFace config.')

    @staticmethod
    def generation_config(parser):
        """Add argument generation_config to parser."""
        return parser.add_argument(
            '--generation-config',
            type=str,
            default='auto',
            help='The folder path to the generation config. Defaults to "auto", the '
            'generation config will be loaded from model path. If set to "lmdeploy", no '
            'generation config is loaded, lmdeploy defaults will be used. If set to a folder '
            'path, the generation config will be loaded from the specified folder path.')

    @staticmethod
    def use_logn_attn(parser):
        """Add argument use_logn_attn to parser."""

        return parser.add_argument('--use-logn-attn',
                                   action='store_true',
                                   default=False,
                                   help='Whether to use logn attention scaling')

    @staticmethod
    def top_p(parser):
        """Add argument top_p to parser."""

        return parser.add_argument('--top-p',
                                   type=float,
                                   default=0.8,
                                   help='An alternative to sampling with temperature,'
                                   ' called nucleus sampling, where the model '
                                   'considers the results of the tokens with '
                                   'top_p probability mass')

    @staticmethod
    def top_k(parser):
        """Add argument top_k to parser."""

        return parser.add_argument('--top-k',
                                   type=int,
                                   default=1,
                                   help='An alternative to sampling with temperature, '
                                   'where the model considers the top_k tokens '
                                   'with the highest probability')

    @staticmethod
    def temperature(parser, default: float = 0.8):
        return parser.add_argument('-temp', '--temperature', type=float, default=default, help='Sampling temperature')

    @staticmethod
    def repetition_penalty(parser):
        """Add argument repetition_penalty to parser."""

        return parser.add_argument('--repetition-penalty',
                                   type=float,
                                   default=1.0,
                                   help='Parameter to penalize repetition')

    @staticmethod
    def log_level(parser):
        """Add argument log_level to parser."""

        import logging
        return parser.add_argument('--log-level',
                                   type=str,
                                   default='WARNING',
                                   choices=list(logging._nameToLevel.keys()),
                                   help='Set the log level')

    @staticmethod
    def api_keys(parser):
        return parser.add_argument(
            '--api-keys',
            type=str,
            nargs='*',
            default=None,
            help='Optional list of space separated API keys',
        )

    @staticmethod
    def ssl(parser):
        return parser.add_argument(
            '--ssl',
            action='store_true',
            required=False,
            default=False,
            help='Enable SSL. Requires OS Environment variables'
            " 'SSL_KEYFILE' and 'SSL_CERTFILE'",
        )

    @staticmethod
    def backend(parser):
        """Add argument backend to parser."""

        def _parse(x):
            if x != 'turbomind':
                raise argparse.ArgumentTypeError(f'unsupported backend {x!r}: TurboMind is the only inference '
                                                 'backend; the PyTorch engine has been removed')
            return x

        return parser.add_argument('--backend',
                                   type=_parse,
                                   default='turbomind',
                                   help='The inference backend. Only "turbomind" is supported')

    @staticmethod
    def stream_output(parser):
        """Add argument stream_output to parser."""

        return parser.add_argument('--stream-output', action='store_true', help='Indicator for streaming output or not')

    @staticmethod
    def calib_dataset(parser):
        """Add argument calib_dataset to parser."""

        return parser.add_argument(
            '--calib-dataset',
            type=str,
            default='wikitext2',
            choices=['wikitext2', 'c4', 'pileval', 'gsm8k', 'neuralmagic_calibration', 'open-platypus', 'openwebtext'],
            help='The calibration dataset name.')

    @staticmethod
    def calib_samples(parser):
        """Add argument calib_samples to parser."""

        return parser.add_argument('--calib-samples',
                                   type=int,
                                   default=128,
                                   help='The number of samples for calibration. '
                                        'Define 0 to indicate the data free quantization.')

    @staticmethod
    def calib_seqlen(parser):
        """Add argument calib_seqlen to parser."""

        return parser.add_argument('--calib-seqlen', type=int, default=2048, help='The sequence length for calibration')

    @staticmethod
    def calib_batchsize(parser):
        """Add argument batch_size to parser."""

        return parser.add_argument(
            '--batch-size',
            type=int,
            default=1,
            help=\
            'The batch size for running the calib samples. Low GPU mem requires small batch_size. Large batch_size reduces the calibration time while costs more VRAM'  # noqa
        )

    @staticmethod
    def calib_search_scale(parser):
        """Add argument search_scale to parser."""

        return parser.add_argument(
            '--search-scale',
            action='store_true',
            default=False,
            help=\
            'Whether search scale ratio. Default to be disabled, which means only smooth quant with 0.5 ratio will be applied'  # noqa
        )

    @staticmethod
    def chat_template(parser):
        """Add chat template config to parser."""

        return parser.add_argument(
            '--chat-template',
            type=str,
            default=None,
            help=\
            'A JSON file or string that specifies the chat template configuration. '  # noqa
            'Please refer to https://lmdeploy.readthedocs.io/en/latest/advance/chat_template.html for the specification'  # noqa
        )

    @staticmethod
    def reasoning_parser(parser):
        """Add reasoning parser to parser."""
        from lmdeploy.serve.parsers.reasoning_parser import LEGACY_REASONING_PARSER_NAMES, ReasoningParserManager
        return parser.add_argument(
            '--reasoning-parser',
            type=str,
            default=None,
            help=f'The registered reasoning parser name: {ReasoningParserManager.module_dict.keys()}. '
            f'Legacy names: {list(LEGACY_REASONING_PARSER_NAMES)}. '
            'Default to None.')

    @staticmethod
    def tool_call_parser(parser):
        """Add tool call parser to parser."""
        from lmdeploy.serve.parsers.tool_parser import ToolParserManager

        return parser.add_argument(
            '--tool-call-parser',
            type=str,
            default=None,
            help=f'The registered tool parser name {ToolParserManager.module_dict.keys()}. Default to None.')

    @staticmethod
    def allow_terminate_by_client(parser):
        """Add argument allow_terminate_by_client to parser."""

        return parser.add_argument('--allow-terminate-by-client',
                                   action='store_true',
                                   default=False,
                                   help='Enable server to be terminated by request from client')

    @staticmethod
    def enable_abort_handling(parser):
        """Add --enable-abort-handling argument to configure server abort
        request processing."""

        return parser.add_argument('--enable-abort-handling',
                                   action='store_true',
                                   default=False,
                                   help='Enable server to handle client abort requests')

    @staticmethod
    def cache_max_entry_count(parser):
        """Add argument cache_max_entry_count to parser."""

        return parser.add_argument('--cache-max-entry-count',
                                   type=float,
                                   default=0.8,
                                   help='The percentage of free gpu memory occupied by the k/v '
                                   'cache, excluding weights ')

    @staticmethod
    def work_dir(parser):
        """Add argument work_dir to parser."""

        return parser.add_argument('--work-dir',
                                   type=str,
                                   default='./work_dir',
                                   help='The working directory to save results')

    @staticmethod
    def cache_block_seq_len(parser):
        """Add argument cache_block_seq_len to parser."""

        return parser.add_argument('--cache-block-seq-len',
                                   type=int,
                                   default=64,
                                   help='The length of the token sequence in a k/v block. '
                                   'If the GPU compute capability is >= 8.0, it should be a '
                                   'multiple of 32, otherwise it should be a multiple of 64')

    @staticmethod
    def enable_prefix_caching(parser):
        """Add argument enable_prefix_caching to parser."""

        return parser.add_argument('--enable-prefix-caching',
                                   action='store_true',
                                   default=False,
                                   help='Enable cache and match prefix')

    @staticmethod
    def num_tokens_per_iter(parser):
        return parser.add_argument('--num-tokens-per-iter',
                                   type=int,
                                   default=0,
                                   help='the number of tokens processed in a forward pass')

    @staticmethod
    def max_prefill_iters(parser):
        return parser.add_argument('--max-prefill-iters',
                                   type=int,
                                   default=1,
                                   help='the max number of forward passes in prefill stage')

    @staticmethod
    def async_(parser):
        return parser.add_argument('--async',
                                   type=int,
                                   default=1,
                                   choices=[0, 1],
                                   dest='async_',
                                   help='Enable async execution (default: 1, enabled). '
                                   'Set to 0 to disable async mode, 1 to enable it.')

    @staticmethod
    def max_prefill_token_num(parser):
        return parser.add_argument('--max-prefill-token-num',
                                   type=int,
                                   default=8192,
                                   help='the max number of tokens per iteration during prefill')

    @staticmethod
    def max_log_len(parser):
        return parser.add_argument('--max-log-len',
                                   type=int,
                                   default=None,
                                   help='Max number of prompt characters or prompt tokens being '
                                   'printed in log. Default: Unlimited')

    @staticmethod
    def disable_fastapi_docs(parser):
        return parser.add_argument('--disable-fastapi-docs',
                                   action='store_true',
                                   default=False,
                                   help="Disable FastAPI's OpenAPI schema,"
                                   ' Swagger UI, and ReDoc endpoint')

    @staticmethod
    def communicator(parser):
        return parser.add_argument('--communicator',
                                   type=str,
                                   default='nccl',
                                   choices=['nccl', 'native', 'cuda-ipc'],
                                   help='Communication backend for multi-GPU inference. The "native" option is '
                                   'deprecated and serves as an alias for "cuda-ipc"')

    @staticmethod
    def moe_a2a_backend(parser):
        return parser.add_argument('--moe-a2a-backend',
                                   type=str,
                                   default='auto',
                                   choices=['auto', 'default', 'deepep'],
                                   help='Communication backend for MoE all-to-all. The "auto" option selects '
                                   '"default" for single-node inference and "deepep" for multi-node inference')

    @staticmethod
    def disable_metrics(parser):
        """Add argument disable_metrics to parser."""
        return parser.add_argument('--disable-metrics',
                                   action='store_true',
                                   default=False,
                                   help='disable metrics system')

    @staticmethod
    def language_model_only(parser):
        """Run as text-only LLM without loading vision/multimodal encoder."""
        return parser.add_argument('--language-model-only',
                                   action='store_true',
                                   default=False,
                                   help='Run as text-only LLM: do not load vision/multimodal encoder modules.')

    @staticmethod
    def add_spec_group(parser):
        spec_group = parser.add_argument_group('Speculative decoding arguments')
        spec_group.add_argument('--speculative-algorithm',
                                type=str,
                                default=None,
                                choices=['mtp', 'eagle3'],
                                help='The speculative algorithm to use. `None` means speculative decoding is disabled')

        spec_group.add_argument('--speculative-draft-model',
                                type=str,
                                default=None,
                                help='The path to speculative draft model. Required by eagle3')

        spec_group.add_argument('--speculative-num-draft-tokens',
                                type=int,
                                default=1,
                                help='The number of speculative tokens to generate per step')

        return spec_group

    @staticmethod
    def trust_remote_code(parser):
        """Add argument trust_remote_code to parser."""
        return parser.add_argument('--trust-remote-code',
                                   action='store_true',
                                   default=False,
                                   help='Whether to trust remote code from model repositories.')


# adapted from https://github.com/vllm-project/vllm/blob/main/vllm/utils/__init__.py
class FlexibleArgumentParser(argparse.ArgumentParser):
    """"More flexible argument parser."""

    def parse_args(self, args=None, namespace=None):
        # If args is not provided, use arguments from the command line
        if args is None:
            args = sys.argv[1:]

        def repl(match: re.Match) -> str:
            """Replaces underscores with dashes in the matched string."""
            return match.group(0).replace('_', '-')

        # Everything between the first -- and the first .
        pattern = re.compile(r'(?<=--)[^\.]*')

        # Convert underscores to dashes and vice versa in argument names
        processed_args = []
        for arg in args:
            if arg.startswith('--'):
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    key = pattern.sub(repl, key, count=1)
                    processed_args.append(f'{key}={value}')
                else:
                    key = pattern.sub(repl, arg, count=1)
                    processed_args.append(key)
            elif arg.startswith('-O') and arg != '-O' and len(arg) == 2:
                # allow -O flag to be used without space, e.g. -O3
                processed_args.append('-O')
                processed_args.append(arg[2:])
            else:
                processed_args.append(arg)

        def _try_convert(value: str):
            """Try to convert string to float or int."""
            if not isinstance(value, str):
                return value
            # try loads from json
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
            return value

        def create_nested_dict(keys: list[str], value: str):
            """Creates a nested dictionary from a list of keys and a value.

            For example, `keys = ["a", "b", "c"]` and `value = 1` will create: `{"a": {"b": {"c": 1}}}`
            """
            nested_dict: Any = _try_convert(value)
            for key in reversed(keys):
                nested_dict = {key: nested_dict}
            return nested_dict

        def recursive_dict_update(original: dict, update: dict):
            """Recursively updates a dictionary with another dictionary."""
            for k, v in update.items():
                if isinstance(v, dict) and isinstance(original.get(k), dict):
                    recursive_dict_update(original[k], v)
                else:
                    original[k] = v

        delete = set()
        dict_args: dict[str, dict] = defaultdict(dict)
        for i, processed_arg in enumerate(processed_args):
            if processed_arg.startswith('--') and '.' in processed_arg:
                if '=' in processed_arg:
                    processed_arg, value = processed_arg.split('=', 1)
                    if '.' not in processed_arg:
                        # False positive, . was only in the value
                        continue
                else:
                    value = processed_args[i + 1]
                    delete.add(i + 1)
                key, *keys = processed_arg.split('.')
                # Merge all values with the same key into a single dict
                arg_dict = create_nested_dict(keys, value)
                recursive_dict_update(dict_args[key], arg_dict)
                delete.add(i)
        # Filter out the dict args we set to None
        processed_args = [a for i, a in enumerate(processed_args) if i not in delete]
        # Add the dict args back as if they were originally passed as JSON
        for dict_arg, dict_value in dict_args.items():
            processed_args.append(dict_arg)
            processed_args.append(json.dumps(dict_value))

        return super().parse_args(processed_args, namespace)
