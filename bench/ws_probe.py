#!/usr/bin/env python3
"""Feed the /ws/transcribe socket the way the browser test page feeds it.

MediaRecorder.start(1000) emits one header-bearing first chunk and then bare
continuation fragments. This reproduces that shape from a webm file so the
server sees exactly what a real recording session sends.

  uv run python bench/ws_probe.py ws://127.0.0.1:8879/ws/transcribe bench/clips/pt_15.wav
"""

import asyncio
import subprocess
import sys
import time
from pathlib import Path

import websockets


async def probe(url: str, wav: Path, chunk_seconds: float = 1.0) -> None:
    webm = wav.with_suffix(".probe.webm")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
         "-c:a", "libopus", "-b:a", "32k", str(webm)],
        check=True,
    )
    blob = webm.read_bytes()
    # Split by size the way a 1-second timeslice roughly would.
    duration = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(wav)],
        capture_output=True, text=True, check=True).stdout)
    n = max(1, round(duration / chunk_seconds))
    size = len(blob) // n
    chunks = [blob[i * size:(i + 1) * size] for i in range(n - 1)] + [blob[(n - 1) * size:]]

    async with websockets.connect(url, max_size=None) as ws:
        for i, chunk in enumerate(chunks):
            t0 = time.perf_counter()
            await ws.send(chunk)
            reply = await asyncio.wait_for(ws.recv(), timeout=120)
            print(f"chunk {i} ({len(chunk)}B) {time.perf_counter()-t0:.2f}s -> {reply[:220]}")
    webm.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(probe(sys.argv[1], Path(sys.argv[2])))
