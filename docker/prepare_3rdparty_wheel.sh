#!/bin/bash -ex

export PATH=/opt/py3/bin:$PATH

pip install "cmake<4.0" wheel ninja setuptools packaging

mkdir -p /wheels

if [[ ${PYTHON_VERSION} = "3.13" ]]; then
    curl https://sh.rustup.rs -sSf | sh -s -- -y
    . "$HOME/.cargo/env"

    pip install setuptools_rust
    pip wheel -v --no-build-isolation --no-deps -w /wheels "git+https://github.com/google/sentencepiece.git@v0.2.0#subdirectory=python"
fi
