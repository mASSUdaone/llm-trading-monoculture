#!/usr/bin/env python3
"""
generate_gemini.py -- generate trading agents from prompt.txt via Gemini.

    python3 generate_gemini.py                 # writes agents/gemini_001.py
    python3 generate_gemini.py --n 5
    python3 generate_gemini.py --start 101 --n 100

Same hygiene as the Anthropic version, so the two populations are comparable:
  * one user message, containing prompt.txt verbatim
  * no system instruction, no tools, no prior turns, no follow-ups
  * pinned model ID, fixed temperature, fixed max output tokens
  * a truncated response is DISCARDED, not continued

Key: put GOOGLE=... in keys.txt, or set GOOGLE_API_KEY.
Get one free at aistudio.google.com -- no card, no billing account.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time

from google import genai
from google.genai import types

MODEL = "gemini-3.6-flash"
TEMPERATURE = 1.0
MAX_TOKENS = 40000   # Gemini 3 thinking tokens count against this
PROMPT_PATH = "prompt.txt"
OUT_DIR = "agents"
PREFIX = "gemini"
SLEEP = 4.0
RETRIES = 6          # free tier is rate limited per minute; be polite


def load_key(path="keys.txt"):
    k = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if k:
        return k
    if os.path.exists(path):
        for ln in open(path):
            ln = ln.strip()
            up = ln.upper()
            if up.startswith("GOOGLE=") or up.startswith("GEMINI="):
                return ln.split("=", 1)[1].strip()
    print("error: no API key. Add a line to keys.txt:\n  GOOGLE=...", file=sys.stderr)
    sys.exit(1)


def extract_code(text):
    """Pull the Python out of the response. Prefer fenced blocks; take the longest."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    return text.strip() + "\n"


def generate(client, prompt, idx):
    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=TEMPERATURE,
            max_output_tokens=MAX_TOKENS,
        ),
    )
    text = resp.text or ""
    code = extract_code(text)

    finish = None
    try:
        finish = str(resp.candidates[0].finish_reason)
    except Exception:
        pass
    truncated = bool(finish and "MAX_TOKEN" in finish.upper())

    stem = os.path.join(OUT_DIR, f"{PREFIX}_{idx:03d}")
    if truncated:
        # keep the evidence, but write no .py -- a rerun regenerates this slot
        # rather than skipping it, and the agent never enters the population
        open(stem + ".rejected.txt", "w").write(text)
    else:
        open(stem + ".raw.txt", "w").write(text)
        open(stem + ".py", "w").write(code)
    usage = getattr(resp, "usage_metadata", None)

    meta = {
        "index": idx,
        "provider": "google",
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_output_tokens": MAX_TOKENS,
        "finish_reason": finish,
        "input_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
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
    print(f"  {PREFIX}_{idx:03d}  {meta['output_tokens']} tok  "
          f"{len(code):>6} chars  {finish}{flag}")
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

    client = genai.Client(api_key=load_key())
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
                          f"{type(e).__name__}: {str(e)[:100]}")
                else:
                    wait = min(120, 15 * (2 ** attempt))
                    print(f"      retry {attempt + 1} in {wait}s ({type(e).__name__})")
                    time.sleep(wait)
        time.sleep(SLEEP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
