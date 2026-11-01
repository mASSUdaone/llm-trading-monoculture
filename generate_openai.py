#!/usr/bin/env python3
"""
generate_openai.py -- generate trading agents from prompt.txt via OpenAI.

    python3 generate_openai.py                  # writes agents/gpt_001.py
    python3 generate_openai.py --n 30

Same hygiene as the Claude and Gemini generators, so the three populations
are comparable:
  * one user message, containing prompt.txt verbatim
  * no system prompt, no tools, no prior turns, no follow-ups
  * pinned model ID, fixed reasoning effort, fixed max output tokens
  * a truncated response is DISCARDED, not continued

Key: put OPENAI=sk-... in keys.txt, or set OPENAI_API_KEY.

Note on temperature: reasoning models do not accept a temperature parameter,
so this generator does not send one. Claude and Gemini ran at temperature 1.0,
which is their default. All three therefore run at each provider's default
sampling behaviour -- record that in the pre-commitment document rather than
claiming the three are numerically identical.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time

from openai import OpenAI

MODEL = "gpt-5.6-terra"
REASONING_EFFORT = "medium"     # provider default; none/low/medium/high/xhigh/max
MAX_OUTPUT_TOKENS = 32000       # reasoning tokens count against this
PROMPT_PATH = "prompt.txt"
OUT_DIR = "agents"
PREFIX = "gpt"
RETRIES = 6
SLEEP = 0.5


def load_key(path="keys.txt"):
    k = os.getenv("OPENAI_API_KEY")
    if k:
        return k
    if os.path.exists(path):
        for ln in open(path):
            ln = ln.strip()
            if ln.upper().startswith("OPENAI="):
                return ln.split("=", 1)[1].strip()
    print("error: no API key. Add a line to keys.txt:\n  OPENAI=sk-...",
          file=sys.stderr)
    sys.exit(1)


def extract_code(text):
    """Pull the Python out of the response. Prefer fenced blocks; take the longest."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    return text.strip() + "\n"


def generate(client, prompt, idx):
    resp = client.responses.create(
        model=MODEL,
        input=prompt,
        reasoning={"effort": REASONING_EFFORT},
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    text = getattr(resp, "output_text", "") or ""
    status = getattr(resp, "status", None)
    incomplete = getattr(resp, "incomplete_details", None)
    reason = getattr(incomplete, "reason", None) if incomplete else None
    truncated = status == "incomplete" or reason == "max_output_tokens"

    code = extract_code(text)
    stem = os.path.join(OUT_DIR, f"{PREFIX}_{idx:03d}")
    if truncated or not text.strip():
        open(stem + ".rejected.txt", "w").write(text)
    else:
        open(stem + ".raw.txt", "w").write(text)
        open(stem + ".py", "w").write(code)

    usage = getattr(resp, "usage", None)
    out_tok = getattr(usage, "output_tokens", None)
    in_tok = getattr(usage, "input_tokens", None)
    details = getattr(usage, "output_tokens_details", None)
    reasoning_tok = getattr(details, "reasoning_tokens", None) if details else None

    meta = {
        "index": idx,
        "provider": "openai",
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "status": status,
        "incomplete_reason": reason,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "reasoning_tokens": reasoning_tok,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "fenced": "```" in text,
        "chars": len(code),
    }
    json.dump(meta, open(stem + ".meta.json", "w"), indent=2)

    flag = ""
    if truncated:
        flag = "  *** TRUNCATED -- no .py written, rerun to regenerate ***"
    elif not meta["fenced"]:
        flag = "  (no code fence; check the raw output)"
    print(f"  {PREFIX}_{idx:03d}  {out_tok} tok "
          f"({reasoning_tok} reasoning)  {len(code):>6} chars  {status}{flag}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--start", type=int, default=1)
    args = ap.parse_args()

    if not os.path.exists(PROMPT_PATH):
        print(f"error: {PROMPT_PATH} not found", file=sys.stderr)
        return 1
    prompt = open(PROMPT_PATH).read()
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"model {MODEL} | effort {REASONING_EFFORT} | prompt "
          f"{hashlib.sha256(prompt.encode()).hexdigest()[:12]}\n")

    client = OpenAI(api_key=load_key())
    for i in range(args.start, args.start + args.n):
        if os.path.exists(os.path.join(OUT_DIR, f"{PREFIX}_{i:03d}.py")):
            print(f"  {PREFIX}_{i:03d}  exists, skipping")
            continue
        for attempt in range(RETRIES):
            try:
                generate(client, prompt, i)
                break
            except Exception as e:
                if attempt == RETRIES - 1:
                    print(f"  {PREFIX}_{i:03d}  FAILED after {RETRIES} tries: "
                          f"{type(e).__name__}: {str(e)[:120]}")
                else:
                    wait = min(120, 15 * (2 ** attempt))
                    print(f"      retry {attempt + 1} in {wait}s "
                          f"({type(e).__name__}: {str(e)[:100]})")
                    time.sleep(wait)
        time.sleep(SLEEP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
