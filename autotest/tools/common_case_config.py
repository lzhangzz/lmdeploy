
TURBOMIND_PR_TEST_LLM_GPU2 = [{
    'model': 'Qwen/Qwen3-30B-A3B',
    'backend': 'turbomind',
    'communicator': 'cuda-ipc',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}, {
    'model': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    'backend': 'turbomind',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}]

TURBOMIND_PR_TEST_LLM_GPU1 = [{
    'model': 'Qwen/Qwen3-0.6B',
    'backend': 'turbomind',
    'communicator': 'cuda-ipc',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}, {
    'model': 'Qwen/Qwen3-0.6B-inner-4bits',
    'backend': 'turbomind',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}, {
    'model': 'Qwen/Qwen3-8B',
    'backend': 'turbomind',
    'communicator': 'nccl',
    'quant_policy': 8,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}]

TURBOMIND_PR_TEST_MLLM_GPU1 = [{
    'model': 'OpenGVLab/InternVL3-8B',
    'backend': 'turbomind',
    'communicator': 'cuda-ipc',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}, {
    'model': 'OpenGVLab/InternVL3-8B',
    'backend': 'turbomind',
    'communicator': 'nccl',
    'quant_policy': 8,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {}
}]

TURBOMIND_PR_TEST_MLLM_GPU2 = [{
    'model': 'OpenGVLab/InternVL3_5-30B-A3B',
    'backend': 'turbomind',
    'communicator': 'cuda-ipc',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}, {
    'model': 'OpenGVLab/InternVL3_5-30B-A3B',
    'backend': 'turbomind',
    'communicator': 'nccl',
    'quant_policy': 8,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}]

TURBOMIND_LOGPROBS_TEST_LLM_GPU2 = [{
    'model': 'Qwen/Qwen3-30B-A3B',
    'backend': 'turbomind',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}, {
    'model': 'OpenGVLab/InternVL3-38B',
    'backend': 'turbomind',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}]

BASE_MODELSCOPE_CONFIG = [{
    'model': 'Qwen/Qwen2.5-7B-Instruct',
    'communicator': 'cuda-ipc',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {},
    'env': {
        'LMDEPLOY_USE_MODELSCOPE': 'True'
    }
}, {
    'model': 'Qwen/Qwen2.5-7B-Instruct',
    'communicator': 'nccl',
    'quant_policy': 8,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {},
    'env': {
        'LMDEPLOY_USE_MODELSCOPE': 'True'
    }
}]

MODELSCOPE_CONFIG = [{
    **item, 'backend': 'turbomind'
} for item in BASE_MODELSCOPE_CONFIG]

BASE_TOOLCALL_TEST_LLM = [{
    'model': 'Qwen/Qwen3-8B',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 1
    },
    'extra_params': {
        'tool-call-parser': 'qwen'
    }
}, {
    'model': 'meta-llama/Meta-Llama-3-1-70B-Instruct',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 4
    },
    'extra_params': {
        'tool-call-parser': 'llama3'
    }
}, {
    'model': 'Qwen/Qwen3-30B-A3B',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {
        'tool-call-parser': 'qwen'
    }
}]

BASE_REASONING_TEST_LLM = [{
    'model': 'Qwen/Qwen3-VL-30B-A3B-Instruct',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {
        'reasoning-parser': 'qwen-qwq'
    }
}, {
    'model': 'Qwen/Qwen3-30B-A3B',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {
        'reasoning-parser': 'qwen-qwq'
    }
}]

TOOLCALL_TEST_LLM = [{
    **item, 'backend': 'turbomind'
} for item in BASE_TOOLCALL_TEST_LLM]

REASONING_TEST_LLM = [{
    **item, 'backend': 'turbomind'
} for item in BASE_REASONING_TEST_LLM]
