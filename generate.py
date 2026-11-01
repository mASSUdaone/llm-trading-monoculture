#!/usr/bin/env python3
"""
generate.py -- generate one trading agent from prompt.txt in a single API call.

    python3 generate.py                  # writes agents/agent_001.py
    python3 generate.py --n 5            # generates 5 agents
    python3 generate.py --start 101 --n 100

Hygiene, enforced here so it is identical for every agent in the population:
  * one user message, containing prompt.txt verbatim
  * no system prompt, no tools, no prior turns, no follow-ups
  * pinned model ID, fixed temperature, fixed max_tokens
  * a truncated response is DISCARDED, not continued -- continuing it would
    give that agent a conversation history the others do not have

Keys: put ANTHROPIC=sk-ant-... in keys.txt, or set ANTHROPIC_API_KEY.

Every call writes three files into agents/:
    agent_NNN.py        the extracted code
    agent_NNN.raw.txt   the full untouched response
    agent_NNN.meta.json model, temperature, token counts, stop_reason, hash
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time

import anthropic

MODEL = "claude-sonnet-5"   # dateless IDs are pinned snapshots for gen 4.6+
TEMPERATURE = 1.0
MAX_TOKENS = 32000
PROMPT_PATH = "prompt.txt"
RETRIES = 6
OUT_DIR = "agents"


def load_key(path="keys.txt"):
    k = os.getenv("ANTHROPIC_API_KEY")
    if k:
        return k
    if os.path.exists(path):
        for ln in open(path):
            ln = ln.strip()
            if ln.upper().startswith("ANTHROPIC=") or ln.upper().startswith("ANTHROPIC_API_KEY="):
                return ln.split("=", 1)[1].strip()
    print("error: no API key. Add a line to keys.txt:\n  ANTHROPIC=sk-ant-...",
          file=sys.stderr)
    sys.exit(1)


def extract_code(text):
    """Pull the Python out of the response. Prefer fenced blocks; take the longest."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    return text.strip() + "\n"   # unfenced: keep everything, flag at review


def generate(client, prompt, idx):
    # Streamed: the SDK refuses a non-streaming call when max_tokens is high
    # enough that the request could outlive its timeout. get_final_message()
    # returns the same object a non-streaming call would.
    with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        resp = stream.get_final_message()
    text = "".join(b.text for b in resp.content if b.type == "text")
    code = extract_code(text)
    truncated = resp.stop_reason == "max_tokens"

    stem = os.path.join(OUT_DIR, f"agent_{idx:03d}")
    if truncated:
        # keep the evidence, write no .py -- a rerun regenerates this slot
        # rather than skipping it with a broken agent in place
        open(stem + ".rejected.txt", "w").write(text)
    else:
        open(stem + ".raw.txt", "w").write(text)
        open(stem + ".py", "w").write(code)

    meta = {
        "index": idx,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stop_reason": resp.stop_reason,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
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
    print(f"  agent_{idx:03d}  {resp.usage.output_tokens:>6} tok  "
          f"{len(code):>6} chars  {resp.stop_reason}{flag}")
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

    print(f"model {MODEL} | temp {TEMPERATURE} | prompt "
          f"{hashlib.sha256(prompt.encode()).hexdigest()[:12]}\n")

    client = anthropic.Anthropic(api_key=load_key())
    for i in range(args.start, args.start + args.n):
        if os.path.exists(os.path.join(OUT_DIR, f"agent_{i:03d}.py")):
            print(f"  agent_{i:03d}  exists, skipping")
            continue
        for attempt in range(RETRIES):
            try:
                generate(client, prompt, i)
                break
            except Exception as e:
                if attempt == RETRIES - 1:
                    print(f"  agent_{i:03d}  FAILED after {RETRIES} tries: "
                          f"{type(e).__name__}: {str(e)[:100]}")
                else:
                    wait = min(120, 15 * (2 ** attempt))
                    print(f"      retry {attempt + 1} in {wait}s "
                          f"({type(e).__name__}: {str(e)[:120]})")
                    time.sleep(wait)
        time.sleep(0.5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
