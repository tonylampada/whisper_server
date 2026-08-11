#!/usr/bin/env python3
"""Word error rate of each benchmark row against the text fed to the TTS.

Comparison is accent- and punctuation-insensitive and spells small digits out,
because "3 segundos" and "tres segundos" are the same transcription being right.
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from make_clips import CLIPS  # noqa: E402

DIGITS = {"1": "um", "2": "dois", "3": "tres", "4": "quatro", "5": "cinco"}
DIGITS_EN = {"1": "one", "2": "two", "3": "three", "4": "four", "5": "five"}


def norm(text: str, lang: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    words = []
    for w in re.findall(r"[a-z0-9]+", text):
        table = DIGITS if lang == "pt" else DIGITS_EN
        words.append(table.get(w, w))
    return words


def wer(ref: list[str], hyp: list[str]) -> tuple[int, int]:
    """Levenshtein distance over words -> (errors, reference length)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1], len(ref)


if __name__ == "__main__":
    refs = {name: (lang, text) for name, (_v, lang, text) in CLIPS.items()}
    rows = [json.loads(l) for l in (HERE / "results.jsonl").open()]

    per_config = {}
    for r in rows:
        if "error" in r or r.get("clip") not in refs:
            continue
        lang, ref_text = refs[r["clip"]]
        err, n = wer(norm(ref_text, lang), norm(r["text"], lang))
        r["wer"] = round(100 * err / n, 1)
        per_config.setdefault(r["config"], []).append((err, n))
        print(f"{r['config']:26s} {r['clip']:6s} WER {r['wer']:5.1f}%  "
              f"first {r['first_s']}s total {r['total_s']}s")

    print("\n-- aggregate WER (pt / en / all)")
    for cfg, pairs in per_config.items():
        e = sum(p[0] for p in pairs)
        n = sum(p[1] for p in pairs)
        print(f"{cfg:26s} {100*e/n:5.1f}%")

    (HERE / "scored.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows if "wer" in r)
    )
