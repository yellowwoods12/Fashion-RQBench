"""In-process Qwen3-8B generation for the Fashion-RQ benchmark pipeline.

This replaces the Gemini API backend used by the original benchmark scripts.
Qwen3-8B is loaded once per process (bfloat16, single GPU) and reused for every
call, so a full run pays the ~16GB load cost a single time.

Qwen3's hybrid reasoning is switched off on every call. The pipeline parses JSON
straight out of the completion, and an enabled `` thinking`` block would both burn
the token budget and break that parse.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Sequence

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen3-8B")
DEVICE = os.environ.get("QWEN_DEVICE", "cuda:0")

# Qwen3-8B ships with a 40,960-token context and no rope scaling. The benchmark
# prompts built from the largest SID clusters run to ~56k tokens, so the stage-1
# generator needs the model's officially supported YaRN extension. Qwen recommends
# turning it on only when sequences routinely exceed the native window, which is
# the case here; set QWEN_YARN_FACTOR=0 to load at native settings instead.
NATIVE_CONTEXT = 40960
YARN_FACTOR = float(os.environ.get("QWEN_YARN_FACTOR", "0"))

_tokenizer = None
_model = None
_context_limit = NATIVE_CONTEXT


def context_limit() -> int:
    return _context_limit


def _load():
    global _tokenizer, _model, _context_limit
    if _model is not None:
        return _tokenizer, _model

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(MODEL_ID)

    kwargs = {}
    if YARN_FACTOR > 1:
        # transformers 5.x carries RoPE settings in config.rope_parameters (an
        # older release used config.rope_scaling). Merge rather than replace so
        # the model's own rope_theta survives.
        if hasattr(config, "rope_parameters"):
            rope = dict(config.rope_parameters or {})
            rope.update(
                rope_type="yarn",
                factor=YARN_FACTOR,
                original_max_position_embeddings=NATIVE_CONTEXT,
            )
            kwargs["rope_parameters"] = rope
        else:
            kwargs["rope_scaling"] = {
                "rope_type": "yarn",
                "factor": YARN_FACTOR,
                "original_max_position_embeddings": NATIVE_CONTEXT,
            }

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map=DEVICE, **kwargs
        )
    except TypeError:
        # transformers <= 4.55 spells this argument torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE, **kwargs
        )
    model.eval()

    if YARN_FACTOR > 1:
        _context_limit = int(NATIVE_CONTEXT * YARN_FACTOR)
        # The config field is not rewritten by the rope-scaling path, so generate()
        # would warn about the stale 40,960 limit on every long prompt.
        model.config.max_position_embeddings = _context_limit

    _tokenizer, _model = tokenizer, model
    return tokenizer, model


def render_chat(tokenizer, prompt: str, system: Optional[str] = None) -> str:
    """Render a single-turn chat with reasoning disabled."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        # Older templates reject the kwarg. The /no_think soft switch keeps Qwen3
        # out of its reasoning block so the completion stays parseable JSON.
        messages[-1]["content"] = messages[-1]["content"] + "\n\n/no_think"
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def strip_reasoning(text: str) -> str:
    """Drop any  thinking...</think> span the model emitted anyway."""
    return re.sub(r" thinking.*?</think>", "", text, flags=re.DOTALL).strip()


def generate(
    prompts: Sequence[str],
    *,
    system: Optional[str] = None,
    max_new_tokens: int = 2048,
    temperature: float = 0.2,
    top_p: float = 0.8,
    batch_size: int = 4,
) -> List[str]:
    """Return one completion per prompt, with reasoning spans stripped.

    Prompts are grouped into batches of *batch_size*; left padding keeps the
    generated continuation at a fixed offset, so completions are sliced off with
    a single prompt width per batch.
    """
    if not prompts:
        return []

    tokenizer, model = _load()
    device = next(model.parameters()).device
    completions: List[str] = []

    for start in range(0, len(prompts), batch_size):
        chunk = list(prompts[start : start + batch_size])
        rendered = [render_chat(tokenizer, prompt, system) for prompt in chunk]
        inputs = tokenizer(
            rendered, return_tensors="pt", padding=True, add_special_tokens=False
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        width = inputs["input_ids"].shape[1]
        if width + max_new_tokens > _context_limit:
            print(
                f"  [CTX] prompt {width:,} + {max_new_tokens:,} new tokens exceeds the "
                f"{_context_limit:,}-token context window; output will be cut short. "
                "Load with QWEN_YARN_FACTOR>1 to extend it."
            )

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_width = inputs["input_ids"].shape[1]
        for row in outputs:
            decoded = tokenizer.decode(row[prompt_width:], skip_special_tokens=True)
            completions.append(strip_reasoning(decoded))

    return completions