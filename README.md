# whisper-server

HTTP transcription service on [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
It is the speech-to-text engine behind Bridge Commander's microphone and call mode, and works
standalone for anything that can POST a file.

- **`POST /transcribe`** — transcribe an uploaded audio/video file
- **`WS /ws/transcribe`** — streaming: send chunks, get the transcript-so-far back
- **`GET /health`** — `{"status": "ok", "model": "..."}` once the model is loaded
- **`GET /`** — a test page (file upload, microphone, live streaming)

## Requirements

- Linux with an **NVIDIA GPU** and a working driver (`nvidia-smi` answers). Without one it
  falls back to CPU int8 — it works, just several times slower (see `bench/RESULTS.md`).
- **CUDA libraries come from pip**, not from a system CUDA install: `nvidia-cublas-cu12` and
  `nvidia-cudnn-cu12` are dependencies, and `cuda_libs.py` preloads them so CTranslate2 finds
  them without `LD_LIBRARY_PATH`. The driver itself must support CUDA 12.
- [`uv`](https://docs.astral.sh/uv/) and Python ≥ 3.11 (uv fetches it if missing).
- `ffmpeg` is **not** needed: faster-whisper decodes through PyAV.
- Disk: the default model, `large-v3-turbo`, is a ~1.6 GB download into the Hugging Face
  cache (`~/.cache/huggingface`) on first start. ~2 GB VRAM at float16. No HF token needed.

## Install and run

```bash
git clone https://github.com/tonylampada/whisper_server.git
cd whisper_server
uv sync
uv run python server.py          # http://127.0.0.1:8878, first start downloads the model
```

Check it:

```bash
curl -s http://127.0.0.1:8878/health
# {"status":"ok","model":"large-v3-turbo"}
```

The startup log says which device it picked: `Loading Whisper model 'large-v3-turbo' on
cuda/float16...`. If it says `cpu/int8` on a GPU box, the GPU was not reachable — run
`nvidia-smi` and check the CUDA wheels installed (`uv sync`).

## Configuration

All by environment variable:

| Variable | Default | Meaning |
|---|---|---|
| `WHISPER_HOST` | `127.0.0.1` | bind address. `0.0.0.0` is refused — see exposure below |
| `WHISPER_PORT` | `8878` | port |
| `WHISPER_MODEL` | `large-v3-turbo` | `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo` |
| `WHISPER_DEVICE` | auto | `cpu` forces the CPU path |

`large-v3-turbo` on CUDA float16 is both the fastest and the most accurate configuration
measured on Portuguese — `bench/RESULTS.md` has the numbers.

## Exposure

There is **no authentication**. Whoever reaches the port can use it, so the bind address is
the auth boundary:

- Default: loopback. Right when the only client (e.g. a Bridge Commander server) is on the
  same machine.
- Another device must reach it: bind a private-network address, e.g. your Tailscale IP
  (`WHISPER_HOST=100.x.y.z`).
- Never every interface (`0.0.0.0`); the server refuses it.

## API

### Batch

```bash
curl -s http://127.0.0.1:8878/transcribe \
  -F file=@audio.webm \
  -F language=pt
```

| Field | Required | Meaning |
|---|---|---|
| `file` | yes | audio or video: wav, mp3, flac, ogg, webm, m4a, mp4, mkv, mov… |
| `language` | no | `pt`, `en`, …; absent = auto-detect |
| `hotwords` | no | words to bias recognition toward — e.g. a spoken keyword that ends a turn (`câmbio`). Passed to faster-whisper's `hotwords` |
| `prompt` | no | Whisper's `initial_prompt`: text the audio is likely to follow. Stronger than `hotwords` and, in noise, it can make Whisper hallucinate the prompt — prefer `hotwords` |

Empty `hotwords` / `prompt` are the same as absent.

```json
{
  "text": "Transcribed text here...",
  "language": "pt",
  "language_probability": 0.98,
  "segments": [{"start": 0.0, "end": 2.5, "text": "Transcribed text here..."}]
}
```

A failure comes back as `{"error": "..."}`.

### Streaming (WebSocket)

Connect to `ws://127.0.0.1:8878/ws/transcribe` and send binary chunks (e.g. `MediaRecorder`
output). After each chunk the server transcribes everything received so far and answers
`{"text": "...", "language": "pt", "partial": true}`. The text frame `reset` drops the buffer
and starts a new utterance.

### CLI

`transcribe-cli <file> [--language pt]` prints the text (needs `curl` and `jq`;
`WHISPER_URL` overrides the address).

## Run it as a systemd user service

`~/.config/systemd/user/whisper-server.service`:

```ini
[Unit]
Description=Whisper Transcription Server
After=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/whisper_server
ExecStart=%h/.local/bin/uv run python server.py
Restart=on-failure
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=WHISPER_HOST=127.0.0.1
# Environment=WHISPER_HOST=100.x.y.z   # your tailnet IP, if another machine is the client

[Install]
WantedBy=default.target
```

Adjust `WorkingDirectory` to your clone and `ExecStart` to `which uv`. Then:

```bash
systemctl --user daemon-reload
systemctl --user enable --now whisper-server
journalctl --user -u whisper-server -f      # watch the model load
loginctl enable-linger "$USER"              # keep it running without a login session
```

## Tests

```bash
uv run pytest -q
```

They stub the model out, so they need no GPU.

## License

MIT — see `LICENSE`. Models are downloaded from Hugging Face under their own licences
(Whisper weights: MIT).
