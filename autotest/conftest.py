import os

import pytest
import yaml
from utils.config_utils import get_config
from utils.proxy_distributed_utils import ProxyDistributedManager
from utils.pytest_layout_utils import all_layout_mark_names


def pytest_configure(config):
    for name in sorted(all_layout_mark_names()):
        config.addinivalue_line(
            'markers',
            f'{name}: autotest parallel layout ({name})',
        )
    config.addinivalue_line(
        'markers',
        'distributed: multi-node runner (proxy), not single-machine layout',
    )

cli_prompt_case_file = 'autotest/chat_prompt_case.yml'
common_prompt_case_file = 'autotest/prompt_case.yml'



@pytest.fixture(scope='session')
def config():
    cfg = get_config()
    device = cfg.get('device')
    if device:
        os.environ.setdefault('DEVICE', device)
    return cfg


@pytest.fixture(scope='session')
def cli_case_config():
    case_path = os.path.join(cli_prompt_case_file)
    with open(case_path) as f:
        case_config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return case_config


@pytest.fixture(scope='class', autouse=True)
def common_case_config():
    case_path = os.path.join(common_prompt_case_file)
    with open(case_path) as f:
        case_config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return case_config


@pytest.fixture(scope='session')
def shared_proxy_manager(config):
    manager = ProxyDistributedManager()

    manager.start()
    if manager.is_master:
        print(f'🎯 Master node: LMDeploy Proxy started on {manager.master_addr}:{manager.proxy_port}')
        print('⏳ Waiting for worker nodes to connect...')

    yield manager

    print(f'\n[Final Cleanup] Node {manager.node_rank} performing final resource cleanup...')
    manager.cleanup()
