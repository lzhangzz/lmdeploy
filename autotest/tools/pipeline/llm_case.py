import json
import os

import fire
import yaml

from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline
from lmdeploy.messages import SpeculativeConfig

gen_config = GenerationConfig(max_new_tokens=500, min_new_tokens=10)


def run_pipeline_chat_test(model_path, run_config, cases_path, is_pr_test: bool = False):
    communicator = run_config.get('communicator')
    quant_policy = run_config.get('quant_policy')
    extra_params = run_config.get('extra_params', {})
    parallel_config = run_config.get('parallel_config', {})

    backend_config = TurbomindEngineConfig(communicator=communicator, quant_policy=quant_policy)

    # quant format
    model_lower = model_path.lower()
    if 'w4' in model_lower or '4bits' in model_lower or 'awq' in model_lower:
        backend_config.model_format = 'awq'
    elif 'gptq' in model_lower:
        backend_config.model_format = 'gptq'

    # Parallel config
    for para_key in ('dp', 'ep', 'cp'):
        if para_key in parallel_config:
            setattr(backend_config, para_key, parallel_config[para_key])
    if 'tp' in parallel_config and parallel_config['tp'] > 1:
        backend_config.tp = parallel_config['tp']

    speculative_config = None
    spec_cfg = extra_params.pop('speculative_config', None)
    if isinstance(spec_cfg, dict):
        speculative_config = SpeculativeConfig(**spec_cfg)
    else:
        spec_kwargs = {}
        for src, dst in (
            ('speculative-algorithm', 'method'),
            ('speculative-num-draft-tokens', 'num_speculative_tokens'),
            ('speculative-draft-model', 'model'),
        ):
            if src in extra_params:
                spec_kwargs[dst] = extra_params.pop(src)
        if 'method' in spec_kwargs:
            speculative_config = SpeculativeConfig(**spec_kwargs)

    # Extra params: CLI-style kebab-case keys map to TurbomindEngineConfig attributes.
    # YAML/CLI ``null`` means a bare flag (same as ``--enable-prefix-caching``).
    # Fire may also leave it as the string ``'null'`` after shell escaping.
    flag_true_on_null = {'enable_prefix_caching'}
    for key, value in extra_params.items():
        attr_name = key.replace('-', '_')
        if attr_name in flag_true_on_null and (value is None or value == 'null'):
            value = True
        try:
            setattr(backend_config, attr_name, value)
        except AttributeError:
            print(f"Warning: Cannot set attribute '{attr_name}' on backend_config. Skipping.")

    print('backend_config config: ' + str(backend_config))
    print('speculative_config config: ' + str(speculative_config))
    pipe = pipeline(model_path, backend_config=backend_config, speculative_config=speculative_config,
                    trust_remote_code=True)

    cases_path = os.path.join(cases_path)
    with open(cases_path) as f:
        cases_info = yaml.load(f.read(), Loader=yaml.SafeLoader)

    for case in cases_info.keys():
        if is_pr_test and case != 'memory_test':
            continue
        if case != 'code_testcase' and 'code' in model_path.lower():
            continue
        case_info = cases_info.get(case)

        prompts = []
        response_list = []
        for prompt_detail in case_info:
            prompt = list(prompt_detail.keys())[0]
            prompts.append({'role': 'user', 'content': prompt})
            response = pipe([prompts], gen_config=gen_config, log_level='INFO', max_log_len=10)[0].text
            response_list.append({'prompt': prompt, 'response': response})
            prompts.append({'role': 'assistant', 'content': response})

        print(f'[caseresult {case} start]' + json.dumps(response_list, ensure_ascii=False) +
              f'[caseresult {case} end]\n')

    pipe.close()


if __name__ == '__main__':
    fire.Fire()
