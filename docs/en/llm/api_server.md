# OpenAI Compatible Server

This article primarily discusses the deployment of a single LLM model across multiple GPUs on a single node, providing a service that is compatible with the OpenAI interface, as well as the usage of the service API.
For the sake of convenience, we refer to this service as `api_server`. Regarding parallel services with multiple models, please refer to the guide about [Request Distribution Server](proxy_server.md).

In the following sections, we will first introduce methods for starting the service, choosing the appropriate one based on your application scenario.

Next, we focus on the definition of the service's RESTful API, explore the various ways to interact with the interface, and demonstrate how to try the service through the Swagger UI or LMDeploy CLI tools.

## Launch Service

Take the [internlm2_5-7b-chat](https://huggingface.co/internlm/internlm2_5-7b-chat) model hosted on huggingface hub as an example, you can choose one the following methods to start the service.

### Option 1: Launching with lmdeploy CLI

```shell
lmdeploy serve api_server internlm/internlm2_5-7b-chat --server-port 23333
```

The arguments of `api_server` can be viewed through the command `lmdeploy serve api_server -h`, for instance, `--tp` to set tensor parallelism, `--session-len` to specify the max length of the context window, `--cache-max-entry-count` to adjust the GPU mem ratio for k/v cache etc.

### Option 2: Deploying with docker

With LMDeploy [official docker image](https://hub.docker.com/r/openmmlab/lmdeploy/tags), you can run OpenAI compatible server as follows:

```shell
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    -p 23333:23333 \
    --ipc=host \
    openmmlab/lmdeploy:latest \
    lmdeploy serve api_server internlm/internlm2_5-7b-chat
```

The parameters of `api_server` are the same with that mentioned in "[option 1](#option-1-launching-with-lmdeploy-cli)" section

### Option 3: Deploying to Kubernetes cluster

Connect to a running Kubernetes cluster and deploy the internlm2_5-7b-chat model service with [kubectl](https://kubernetes.io/docs/reference/kubectl/) command-line tool (replace `<your token>` with your huggingface hub token):

```shell
sed 's/{{HUGGING_FACE_HUB_TOKEN}}/<your token>/' k8s/deployment.yaml | kubectl create -f - \
    && kubectl create -f k8s/service.yaml
```

In the example above the model data is placed on the local disk of the node (hostPath). Consider replacing it with high-availability shared storage if multiple replicas are desired, and the storage can be mounted into container using [PersistentVolume](https://kubernetes.io/docs/concepts/storage/persistent-volumes/).

## RESTful API

LMDeploy's RESTful API is compatible with the following three OpenAI interfaces:

- /v1/chat/completions
- /v1/models
- /v1/completions

You can overview and try out the offered RESTful APIs by the website `http://0.0.0.0:23333` as shown in the below image after launching the service successfully.

![swagger_ui](https://github.com/InternLM/lmdeploy/assets/4560679/b891dd90-3ffa-4333-92b2-fb29dffa1459)

If you need to integrate the service into your own projects or products, we recommend the following approach:

### Integrate with `OpenAI`

Here is an example of interaction with the endpoint `v1/chat/completions` service via the openai package.
Before running it, please install the openai package by `pip install openai`

```python
from openai import OpenAI
client = OpenAI(
    api_key='YOUR_API_KEY',
    base_url="http://0.0.0.0:23333/v1"
)
model_name = client.models.list().data[0].id
response = client.chat.completions.create(
  model=model_name,
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": " provide three suggestions about time management"},
  ],
    temperature=0.8,
    top_p=0.8
)
print(response)
```

If you want to use async functions, may try the following example:

```python
import asyncio
from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI(api_key='YOUR_API_KEY',
                         base_url='http://0.0.0.0:23333/v1')
    model_cards = await client.models.list()._get_page()
    response = await client.chat.completions.create(
        model=model_cards.data[0].id,
        messages=[
            {
                'role': 'system',
                'content': 'You are a helpful assistant.'
            },
            {
                'role': 'user',
                'content': ' provide three suggestions about time management'
            },
        ],
        temperature=0.8,
        top_p=0.8)
    print(response)

asyncio.run(main())
```

You can invoke other OpenAI interfaces using similar methods. For more detailed information, please refer to the [OpenAI API guide](https://platform.openai.com/docs/guides/text-generation)

### Tools

May refer to [api_server_tools](./api_server_tools.md).

### Anthropic-Compatible Endpoints

May refer to [api_server_anthropic](./api_server_anthropic.md).

### OpenAI Responses-Compatible Endpoint

May refer to [api_server_responses](./api_server_responses.md).

### Integrate with Java/Golang/Rust

May use [openapi-generator-cli](https://github.com/OpenAPITools/openapi-generator-cli) to convert `http://{server_ip}:{server_port}/openapi.json` to java/rust/golang client.
Here is an example:

```shell
$ docker run -it --rm -v ${PWD}:/local openapitools/openapi-generator-cli generate -i /local/openapi.json -g rust -o /local/rust

$ ls rust/*
rust/Cargo.toml  rust/git_push.sh  rust/README.md

rust/docs:
ChatCompletionRequest.md  EmbeddingsRequest.md  HttpValidationError.md  LocationInner.md  Prompt.md
DefaultApi.md             GenerateRequest.md    Input.md                Messages.md       ValidationError.md

rust/src:
apis  lib.rs  models
```

### Integrate with cURL

cURL is a tool for observing the output of the RESTful APIs.

- list served models `v1/models`

```bash
curl http://{server_ip}:{server_port}/v1/models
```

- chat `v1/chat/completions`

```bash
curl http://{server_ip}:{server_port}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "intern-s2-preview",
    "messages": [{"role": "user", "content": "Hello! How are you?"}]
  }'
```

- text completions `v1/completions`

```shell
curl http://{server_ip}:{server_port}/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "llama",
  "prompt": "two steps to build a house:"
}'
```

## Launch multiple api servers

Following are two steps to launch multiple api servers through torchrun. Just create a python script with the following codes.

1. Launch the proxy server through `lmdeploy serve proxy`. Get the correct proxy server url.
2. Launch the script through `torchrun --nproc_per_node 2 script.py InternLM/internlm2-chat-1_8b --proxy_url http://{proxy_node_name}:{proxy_node_port}`.**Note**: Please do not use `0.0.0.0:8000` here, instead, we input the real ip name, `11.25.34.55:8000` for example.

```python
import os
import socket
from typing import List

import fire


def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def main(model_path: str,
         tp: int = 1,
         proxy_url: str = 'http://0.0.0.0:8000',
         port: int = 23333):
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', -1))
    local_ip = get_host_ip()
    if isinstance(port, List):
        assert len(port) == world_size
        port = port[local_rank]
    else:
        port += local_rank * 10
    if (world_size - local_rank) % tp == 0:
        rank_list = ','.join([str(local_rank + i) for i in range(tp)])
        command = f'CUDA_VISIBLE_DEVICES={rank_list} lmdeploy serve api_server {model_path} '\
                  f'--server-name {local_ip} --server-port {port} --tp {tp} '\
                  f'--proxy-url {proxy_url}'
        print(f'running command: {command}')
        os.system(command)


if __name__ == '__main__':
    fire.Fire(main)
```

## FAQ

1. When user got `"finish_reason":"length"`, it means the session is too long to be continued. The session length can be
   modified by passing `--session-len` to api_server.

2. When OOM appeared at the server side, please reduce the `cache_max_entry_count` of `backend_config` when launching the service.
   For the pipeline interface, the same parameter lives on `TurbomindEngineConfig`:

   ```python
   from lmdeploy import pipeline, TurbomindEngineConfig

   backend_config = TurbomindEngineConfig(cache_max_entry_count=0.2)

   pipe = pipeline('internlm/internlm2_5-7b-chat',
                   backend_config=backend_config)
   response = pipe(['Hi, pls intro yourself', 'Shanghai is'])
   print(response)
   ```

   When the server is launched from the CLI, pass `--cache-max-entry-count` instead. The parameter is explained under
   [Engine configuration](pipeline.md#engine-configuration).

3. Regarding the stop words, we only support characters that encode into a single index. Furthermore, there may be multiple indexes that decode into results containing the stop word. In such cases, if the number of these indexes is too large, we will only use the index encoded by the tokenizer. If you want use a stop symbol that encodes into multiple indexes, you may consider performing string matching on the streaming client side. Once a successful match is found, you can then break out of the streaming loop.

4. To customize a chat template, please refer to [chat_template.md](../advance/chat_template.md).

5. `ModuleNotFoundError: No module named '_turbomind'`.

   `_turbomind` is the compiled TurboMind extension. `pip install lmdeploy` ships one; a source install builds it. If the
   import fails anyway, check two things.

   First, do not run from the root of the lmdeploy source tree. Python then imports the `lmdeploy` package from the
   checkout rather than the installed one, and the checkout contains no built extension. Run from another directory.

   Second, the extension is built for the interpreter that compiled it. If several Python environments are installed,
   make sure the one running the command is the one it was built against. When configuring CMake by hand, pass
   `-DPYTHON_EXECUTABLE=/path/to/python3` and rebuild.

6. `libnccl.so.2 not found`.

   Make sure lmdeploy is installed through `pip install lmdeploy[all]`; the `all` extra pulls in the matching NCCL
   package (`nvidia-nccl-cu12` or `nvidia-nccl-cu13`, `>=2.30.4`). If the library is present but not on the loader
   path, add it:

   ```shell
   # Get the location of the NCCL package, e.g. nvidia-nccl-cu12
   pip show nvidia-nccl-cu12 | grep Location
   # insert the path of "libnccl.so.2" to LD_LIBRARY_PATH
   export LD_LIBRARY_PATH={Location}/nvidia/nccl/lib:$LD_LIBRARY_PATH
   ```

7. Image and video URL fetch timeout.

   The API server fetches `image_url` and `video_url` media over HTTP. The timeouts are configurable through the
   environment variables `LMDEPLOY_IMAGE_FETCH_TIMEOUT` (default 10 seconds) and `LMDEPLOY_VIDEO_FETCH_TIMEOUT`
   (default 30 seconds), read in `lmdeploy/vl/media/connection.py`.
