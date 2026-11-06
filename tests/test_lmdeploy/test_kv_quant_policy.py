# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import pytest
from pydantic_core import ValidationError

from lmdeploy.cli.utils import ArgumentHelper
from lmdeploy.messages import QuantPolicy, TurbomindEngineConfig


@pytest.mark.parametrize('value, expected', [
    ('none', QuantPolicy.NONE),
    ('int4', QuantPolicy.INT4),
    ('INT8', QuantPolicy.INT8),
    ('0', QuantPolicy.NONE),
    ('4', QuantPolicy.INT4),
    ('8', QuantPolicy.INT8),
])
def test_quant_policy_cli_aliases(value, expected):
    parser = argparse.ArgumentParser()
    ArgumentHelper.quant_policy(parser)

    assert parser.parse_args(['--quant-policy', value]).quant_policy == expected


def test_quant_policy_uses_configured_default():
    parser = argparse.ArgumentParser()
    ArgumentHelper.quant_policy(parser, default=QuantPolicy.INT8.value)

    assert parser.parse_args([]).quant_policy == QuantPolicy.INT8.value


@pytest.mark.parametrize('value', ['fp8', 'fp8_e5m2', 'turbo_quant', '16', '17', '42'])
def test_quant_policy_cli_rejects_unsupported_policies(value):
    parser = argparse.ArgumentParser()
    ArgumentHelper.quant_policy(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(['--quant-policy', value])


def test_turbomind_config_normalizes_quant_policy_value():
    config = TurbomindEngineConfig(quant_policy=QuantPolicy.INT4.value)

    assert config.quant_policy == QuantPolicy.INT4


@pytest.mark.parametrize('quant_policy', [16, 17, 42, 99])
def test_turbomind_config_rejects_unsupported_quant_policies(quant_policy):
    with pytest.raises(ValidationError, match='invalid quant_policy'):
        TurbomindEngineConfig(quant_policy=quant_policy)
