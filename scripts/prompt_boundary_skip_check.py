#!/usr/bin/env python3
"""Manual harness: verify TurboMind multi-turn prompt-boundary-skip reuse.

Two sequential chat turns on one pipeline (turn 2 sees turn 1's published blocks).
Turn 1 is a thinking-style prompt; turn 2 extends history with the assistant answer
(thinking block stripped) plus a follow-up user message. Parses C++ LogAccept/LogResume
lines to report turn-2 prefix reuse for a chosen cache_prompt_boundary_skip (K).

Run OUTSIDE the sandbox (needs a GPU):

  python scripts/prompt_boundary_skip_check.py \\
      --model-id Qwen/Qwen3-30B-A3B-Thinking-2507 --cache-dir /path/to/hf/cache \\
      --tp 1 --gpus 0

  python scripts/prompt_boundary_skip_check.py ... --skip 1
  python scripts/prompt_boundary_skip_check.py ... --skip 4

Note: the volatile suffix length is template-dependent. Qwen-family thinking
models re-insert an empty `<think>\n\n</think>\n\n` block for completed assistant
turns, so `<think>` stays shared and the volatile suffix is only 1 token (K=1
suffices). The multi-token-K benefit applies to templates that fully EXCLUDE the
thinking block (no `<think>` echo) in history; choose a model/template
accordingly, or pass `--skip` explicitly.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import re
import sys
import tempfile


def _set_hf_cache(cache_dir: str) -> None:
    import huggingface_hub.constants as hf_constants
    hf_constants.HF_HUB_OFFLINE = 1
    hf_constants.HF_HUB_CACHE = cache_dir


@contextlib.contextmanager
def capture_low_level_output():
    """Redirect fds 1 & 2 to a temp file so C++ TM_LOG output is captured."""
    f = tempfile.NamedTemporaryFile('w+', suffix='.log', delete=False)
    saved_out, saved_err = os.dup(1), os.dup(2)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(f.fileno(), 1)
    os.dup2(f.fileno(), 2)
    try:
        yield f.name
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        f.close()


MATCH_RE = re.compile(r'matched \[0,(\d+)\) \((\d+) blk')
FORKTO_RE = re.compile(r'fork_to@(\d+)')
RESUME_RE = re.compile(r'resume \[0,(\d+)\).*source=(\w+)')


def parse_log(path: str):
    matched, fork_to, resume = [], [], []
    with open(path, 'r', errors='replace') as fh:
        for line in fh:
            if (m := MATCH_RE.search(line)):
                matched.append(tuple(int(x) for x in m.groups()))
            if (m := FORKTO_RE.search(line)):
                fork_to.append(int(m.group(1)))
            if (m := RESUME_RE.search(line)):
                resume.append((int(m.group(1)), m.group(2)))
    return matched, fork_to, resume


def _ids(tokenizer, msgs, **kw):
    out = tokenizer.apply_chat_template(msgs, tokenize=True, **kw)
    if hasattr(out, 'keys'):          # BatchEncoding / dict
        out = out['input_ids']
    if out and isinstance(out[0], list):  # batched nesting
        out = out[0]
    return list(out)


def compute_think_suffix_len(model_id: str, cache_dir: str) -> int:
    _set_hf_cache(cache_dir)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    gen_only = _ids(tokenizer, [{'role': 'user', 'content': 'hi'}],
                    add_generation_prompt=True)
    with_assistant = _ids(tokenizer,
                          [{'role': 'user', 'content': 'hi'},
                           {'role': 'assistant', 'content': 'x'}],
                          add_generation_prompt=False)
    lcp = 0
    for a, b in zip(gen_only, with_assistant):
        if a != b:
            break
        lcp += 1
    suffix_len = len(gen_only) - lcp
    tail = repr(tokenizer.decode(gen_only[lcp:]))
    print(f'think-suffix heuristic: len(gen_only)={len(gen_only)} lcp={lcp} '
          f'suffix_len={suffix_len} tail={tail}')
    return max(1, suffix_len)


def strip_thinking_block(text: str) -> str:
    marker = '</think>'
    idx = text.find(marker)
    if idx >= 0:
        return text[idx + len(marker):].lstrip()
    return text


def run(args) -> int:
    _set_hf_cache(args.cache_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    os.environ['TM_LOG_LEVEL'] = 'INFO'

    computed_suffix = None
    if args.skip < 0:
        computed_suffix = compute_think_suffix_len(args.model_id, args.cache_dir)
        skip = computed_suffix
    else:
        skip = args.skip

    from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline

    engine_config = TurbomindEngineConfig(
        tp=args.tp,
        session_len=8192,
        cache_max_entry_count=0.5,
        enable_prefix_caching=True,
        enable_metrics=False,
        cache_prompt='all',
        cache_generation='all',
        cache_prompt_boundary_skip=skip,
    )
    gen_config = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=False)

    msgs1 = [{'role': 'user', 'content': args.turn1}]
    msgs2 = None
    out1 = out2 = None

    with capture_low_level_output() as log_path:
        with pipeline(args.model_id, backend_config=engine_config, log_level='INFO',
                      trust_remote_code=True) as pipe:
            out1 = pipe(msgs1, gen_config=gen_config)
            answer = out1.text if hasattr(out1, 'text') else str(out1)
            answer_stripped = strip_thinking_block(answer)
            msgs2 = [
                {'role': 'user', 'content': args.turn1},
                {'role': 'assistant', 'content': answer_stripped},
                {'role': 'user', 'content': args.turn2},
            ]
            out2 = pipe(msgs2, gen_config=gen_config)

    matched, fork_to, resume = parse_log(log_path)

    text1 = out1.text if hasattr(out1, 'text') else str(out1)
    text2 = out2.text if hasattr(out2, 'text') else str(out2)
    token_ids2 = getattr(out2, 'token_ids', None)

    print(f'=== prompt-boundary-skip reuse check ===')
    print(f'log: {log_path}')
    print(f'cache_prompt_boundary_skip (chosen K): {skip}')
    print(f'computed think-suffix len: {computed_suffix}')
    print(f'--- turn 1 response ({len(text1)} chars) ---')
    print(text1.strip()[:400])
    print(f'--- turn 2 response ({len(text2)} chars) ---')
    print(text2.strip()[:400])
    if token_ids2 is not None:
        print(f'turn 2 token_ids len: {len(token_ids2)}')
    print(f'matched (M, blk): {matched}')
    print(f'fork_to@ positions: {fork_to}')
    print(f'resume (history, source): {resume}')

    turn2_m = matched[-1][0] if matched else None
    turn2_resume_src = resume[-1][1] if resume else None
    max_fork = max(fork_to) if fork_to else None
    print(f'TURN2 SUMMARY: matched_M={turn2_m} max_fork_to@={max_fork} resume_source={turn2_resume_src}')
    print(f'TURN2 OUTPUT: {text2.strip()}')
    if token_ids2 is not None:
        print(f'TURN2 TOKEN_IDS: {list(token_ids2)}')

    ok = bool(text2.strip()) and bool(matched or resume)
    print('RESULT:', 'OK' if ok else 'FAILED')
    return 0 if ok else 1


def main(argv) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model-id', required=True)
    p.add_argument('--cache-dir', required=True)
    p.add_argument('--tp', type=int, required=True)
    p.add_argument('--gpus', required=True)
    p.add_argument('--skip', type=int, default=-1,
                   help='cache_prompt_boundary_skip; -1 = use computed think-suffix len')
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--turn1', default='What is the capital of France? Think step by step.')
    p.add_argument('--turn2', default='Now what is its population, roughly?')
    return run(p.parse_args(argv[1:]))


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
