# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.utils import get_max_batch_size

from .cli import CLI
from .utils import (
    ArgumentHelper,
    DefaultsAndTypesHelpFormatter,
    convert_args,
    get_chat_template,
    get_speculative_config,
)


class SubCliServe:
    """Serve LLMs and interact on terminal."""
    _help = 'Serve LLMs with openai API'
    _desc = _help
    parser = CLI.subparsers.add_parser(
        'serve',
        help=_help,
        description=_desc,
    )
    subparsers = parser.add_subparsers(title='Commands', description='This group has the following commands:')

    @staticmethod
    def add_parser_api_server():
        """Add parser for api_server command."""
        parser = SubCliServe.subparsers.add_parser('api_server',
                                                   formatter_class=DefaultsAndTypesHelpFormatter,
                                                   description=SubCliServe.api_server.__doc__,
                                                   help=SubCliServe.api_server.__doc__)
        parser.set_defaults(run=SubCliServe.api_server)
        parser.add_argument('model_path',
                            type=str,
                            help='The path of a model. it could be one of the following '
                            'options: - i) a local directory path of a turbomind model'
                            ' which is converted by `lmdeploy convert` command or '
                            'download from ii) and iii). - ii) the model_id of a '
                            'lmdeploy-quantized model hosted inside a model repo on '
                            'huggingface.co, such as "lmdeploy/llama2-chat-70b-4bit",'
                            ' etc. - iii) the model_id of a model hosted inside a model'
                            ' repo on huggingface.co, such as "internlm/internlm2_5-7b-chat",'
                            ' "internlm/Intern-S2-Preview" and so on')
        parser.add_argument('--server-name', type=str, default='0.0.0.0', help='Host ip for serving')
        parser.add_argument('--server-port', type=int, default=23333, help='Server port')
        parser.add_argument('--allow-origins',
                            nargs='+',
                            type=str,
                            default=['*'],
                            help='A list of allowed origins for cors')
        parser.add_argument('--allow-credentials', action='store_true', help='Whether to allow credentials for cors')
        parser.add_argument('--allow-methods',
                            nargs='+',
                            type=str,
                            default=['*'],
                            help='A list of allowed http methods for cors')
        parser.add_argument('--allow-headers',
                            nargs='+',
                            type=str,
                            default=['*'],
                            help='A list of allowed http headers for cors')
        parser.add_argument('--proxy-url', type=str, default=None, help='The proxy url for api server.')
        parser.add_argument('--max-concurrent-requests',
                            type=int,
                            default=None,
                            help='This refers to the number of concurrent requests that '
                            'the server can handle. The server is designed to process the '
                            'engine’s tasks once the maximum number of concurrent requests is '
                            'reached, regardless of any additional requests sent by clients '
                            'concurrently during that time. Default to None.')
        parser.add_argument('--allowed-media-domains',
                            nargs='+',
                            type=str,
                            default=None,
                            help='Exact hostnames allowed for HTTP(S) media URLs.')
        # common args
        ArgumentHelper.backend(parser)
        ArgumentHelper.log_level(parser)
        ArgumentHelper.api_keys(parser)
        ArgumentHelper.ssl(parser)
        ArgumentHelper.model_name(parser)
        ArgumentHelper.max_log_len(parser)
        ArgumentHelper.disable_fastapi_docs(parser)
        ArgumentHelper.allow_terminate_by_client(parser)
        ArgumentHelper.enable_abort_handling(parser)
        ArgumentHelper.trust_remote_code(parser)
        # chat template args
        ArgumentHelper.chat_template(parser)

        # parsers
        ArgumentHelper.tool_call_parser(parser)
        ArgumentHelper.reasoning_parser(parser)

        # model args
        ArgumentHelper.revision(parser)
        ArgumentHelper.download_dir(parser)
        ArgumentHelper.generation_config(parser)

        # engine args
        tb_group = parser.add_argument_group('TurboMind engine arguments')
        ArgumentHelper.language_model_only(tb_group)
        ArgumentHelper.dtype(tb_group)
        ArgumentHelper.tp(tb_group)
        ArgumentHelper.dp(tb_group)
        ArgumentHelper.cp(tb_group)
        ArgumentHelper.ep(tb_group)
        ArgumentHelper.session_len(tb_group)
        ArgumentHelper.max_batch_size(tb_group)
        ArgumentHelper.cache_max_entry_count(tb_group)
        ArgumentHelper.cache_block_seq_len(tb_group)
        ArgumentHelper.enable_prefix_caching(tb_group)
        ArgumentHelper.max_prefill_token_num(tb_group)
        ArgumentHelper.quant_policy(tb_group)
        ArgumentHelper.model_format(tb_group)
        ArgumentHelper.hf_overrides(tb_group)
        ArgumentHelper.disable_metrics(tb_group)
        ArgumentHelper.rope_scaling_factor(tb_group)
        ArgumentHelper.num_tokens_per_iter(tb_group)
        ArgumentHelper.max_prefill_iters(tb_group)
        ArgumentHelper.async_(tb_group)
        ArgumentHelper.communicator(tb_group)
        ArgumentHelper.moe_a2a_backend(tb_group)
        # multi-node serving args
        ArgumentHelper.num_nodes(tb_group)
        ArgumentHelper.node_rank(tb_group)
        ArgumentHelper.dist_init_addr(tb_group)

        # spec decode
        ArgumentHelper.add_spec_group(parser)

    @staticmethod
    def add_parser_proxy():
        """Add parser for proxy server command."""
        parser = SubCliServe.subparsers.add_parser('proxy',
                                                   formatter_class=DefaultsAndTypesHelpFormatter,
                                                   description=SubCliServe.proxy.__doc__,
                                                   help=SubCliServe.proxy.__doc__)
        parser.set_defaults(run=SubCliServe.proxy)
        parser.add_argument('--server-name', type=str, default='0.0.0.0', help='Host ip for proxy serving')
        parser.add_argument('--server-port', type=int, default=8000, help='Server port of the proxy')
        parser.add_argument('--routing-strategy',
                            type=str,
                            choices=['random', 'min_expected_latency', 'min_observed_latency'],
                            default='min_expected_latency',
                            help='the strategy to dispatch requests to nodes')
        parser.add_argument('--disable-cache-status',
                            action='store_true',
                            help='Whether to disable cache status of the '
                            'proxy. If set, the proxy will forget the status '
                            'of the previous time')
        ArgumentHelper.api_keys(parser)
        ArgumentHelper.ssl(parser)
        ArgumentHelper.log_level(parser)

    @staticmethod
    def api_server(args):
        """Serve LLMs with restful api using fastapi."""
        from lmdeploy.messages import TurbomindEngineConfig
        from lmdeploy.serve.openai.api_server import serve as run_api_server

        max_batch_size = args.max_batch_size if args.max_batch_size else get_max_batch_size('cuda')
        backend_config = TurbomindEngineConfig(dtype=args.dtype,
                                               tp=args.tp,
                                               dp=args.dp,
                                               cp=args.cp,
                                               ep=args.ep,
                                               nnodes=args.nnodes,
                                               node_rank=args.node_rank,
                                               dist_init_addr=args.dist_init_addr,
                                               max_batch_size=max_batch_size,
                                               session_len=args.session_len,
                                               model_format=args.model_format,
                                               quant_policy=args.quant_policy,
                                               rope_scaling_factor=args.rope_scaling_factor,
                                               cache_max_entry_count=args.cache_max_entry_count,
                                               cache_block_seq_len=args.cache_block_seq_len,
                                               enable_prefix_caching=args.enable_prefix_caching,
                                               max_prefill_token_num=args.max_prefill_token_num,
                                               num_tokens_per_iter=args.num_tokens_per_iter,
                                               max_prefill_iters=args.max_prefill_iters,
                                               async_=args.async_,
                                               communicator=args.communicator,
                                               moe_a2a_backend=args.moe_a2a_backend,
                                               language_model_only=args.language_model_only,
                                               enable_metrics=not args.disable_metrics,
                                               hf_overrides=args.hf_overrides)
        chat_template_config = get_chat_template(args.chat_template, args.model_path)
        speculative_config = get_speculative_config(args)

        run_api_server(
            args.model_path,
            model_name=args.model_name,
            backend_config=backend_config,
            chat_template_config=chat_template_config,
            server_name=args.server_name,
            server_port=args.server_port,
            allow_origins=args.allow_origins,
            allow_credentials=args.allow_credentials,
            allow_methods=args.allow_methods,
            allow_headers=args.allow_headers,
            allow_terminate_by_client=args.allow_terminate_by_client,
            enable_abort_handling=args.enable_abort_handling,
            log_level=args.log_level.upper(),
            api_keys=args.api_keys,
            ssl=args.ssl,
            proxy_url=args.proxy_url,
            max_log_len=args.max_log_len,
            disable_fastapi_docs=args.disable_fastapi_docs,
            max_concurrent_requests=args.max_concurrent_requests,
            trust_remote_code=args.trust_remote_code,
            reasoning_parser=args.reasoning_parser,
            tool_call_parser=args.tool_call_parser,
            speculative_config=speculative_config,
            allowed_media_domains=args.allowed_media_domains,
            generation_config=args.generation_config,
        )

    @staticmethod
    def proxy(args):
        """Proxy server that manages distributed api_server nodes."""
        from lmdeploy.serve.proxy.proxy import proxy
        kwargs = convert_args(args)
        proxy(**kwargs)

    @staticmethod
    def add_parsers():
        SubCliServe.add_parser_api_server()
        SubCliServe.add_parser_proxy()
