#!/usr/bin/env python3
"""Replay the same clips through every whisper configuration and time it.

Each configuration runs in its own subprocess so one model's VRAM never taints
the next one's numbers. Results are appended to bench/results.jsonl.

  uv run python bench/bench.py                 # every configuration
  uv run python bench/bench.py --only cuda-lv3-fp16
  uv run python bench/bench.py --worker <cfg>  # internal: one config, one process
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
CLIPS_DIR = HERE / "clips"
RESULTS = HERE / "results.jsonl"

CLIPS = ["pt_03", "pt_15", "pt_60", "en_03", "en_15", "en_60"]

# name -> (model, device, compute_type, batched)
CONFIGS = {
    "cpu-base-int8":       ("base",           "cpu",  "int8",         False),
    "cpu-lv3-int8":        ("large-v3",       "cpu",  "int8",         False),
    "cuda-lv3-fp16":       ("large-v3",       "cuda", "float16",      False),
    "cuda-lv3-int8fp16":   ("large-v3",       "cuda", "int8_float16", False),
    "cuda-turbo-fp16":     ("large-v3-turbo", "cuda", "float16",      False),
    "cuda-turbo-int8fp16": ("large-v3-turbo", "cuda", "int8_float16", False),
    "cuda-medium-fp16":    ("medium",         "cuda", "float16",      False),
    "cuda-base-fp16":      ("base",           "cuda", "float16",      False),
    "cuda-turbo-fp16-batched": ("large-v3-turbo", "cuda", "float16",  True),
    "cuda-lv3-fp16-batched":   ("large-v3",       "cuda", "float16",  True),
}

# The server's own transcribe() settings, so the benchmark measures the server
# and not some tuned-up variant of it.
DECODE = dict(beam_size=5, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))


def run_config(cfg_name: str) -> list[dict]:
    sys.path.insert(0, str(HERE.parent))
    import cuda_libs  # noqa: F401  (preloads libcublas/libcudnn for CTranslate2)
    from faster_whisper import WhisperModel, BatchedInferencePipeline

    model_name, device, compute_type, batched = CONFIGS[cfg_name]

    t0 = time.perf_counter()
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    engine = BatchedInferencePipeline(model=model) if batched else model
    load_s = time.perf_counter() - t0

    kwargs = dict(DECODE)
    if batched:
        kwargs["batch_size"] = 8

    # One warm-up pass so the first measured clip is not paying for lazy CUDA
    # kernel init and cuDNN autotuning.
    list(engine.transcribe(str(CLIPS_DIR / "en_03.wav"), language="en", **kwargs)[0])

    rows = []
    for clip in CLIPS:
        path = CLIPS_DIR / f"{clip}.wav"
        lang = clip.split("_")[0]

        t0 = time.perf_counter()
        segments, info = engine.transcribe(str(path), language=lang, **kwargs)
        # faster-whisper yields segments lazily: pulling the first one is the
        # closest thing this API has to time-to-first-token.
        first_s = None
        texts = []
        for seg in segments:
            if first_s is None:
                first_s = time.perf_counter() - t0
            texts.append(seg.text.strip())
        total_s = time.perf_counter() - t0

        rows.append({
            "config": cfg_name,
            "model": model_name,
            "device": device,
            "compute_type": compute_type,
            "batched": batched,
            "clip": clip,
            "load_s": round(load_s, 2),
            "first_s": round(first_s, 3) if first_s is not None else None,
            "total_s": round(total_s, 3),
            "text": " ".join(texts),
        })
        print(f"  {clip:6s} first={first_s if first_s is None else round(first_s,2)}s "
              f"total={total_s:.2f}s", flush=True)

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--worker", default=None)
    args = ap.parse_args()

    if args.worker:
        for row in run_config(args.worker):
            print("ROW " + json.dumps(row), flush=True)
        return

    names = args.only or list(CONFIGS)
    with RESULTS.open("a") as out:
        for name in names:
            print(f"=== {name}", flush=True)
            proc = subprocess.run(
                [sys.executable, __file__, "--worker", name],
                capture_output=True, text=True,
            )
            if proc.returncode != 0:
                print(proc.stdout[-2000:])
                print(proc.stderr[-2000:])
                out.write(json.dumps({"config": name, "error": proc.stderr[-1500:]}) + "\n")
                out.flush()
                continue
            for line in proc.stdout.splitlines():
                if line.startswith("ROW "):
                    out.write(line[4:] + "\n")
                else:
                    print(line)
            out.flush()


if __name__ == "__main__":
    main()
