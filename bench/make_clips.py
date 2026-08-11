#!/usr/bin/env python3
"""Synthesize the benchmark clips from the local VoxCPM2 TTS server (port 8883).

Read-only use of that server: POST /v1/audio/speech, nothing restarted or reconfigured.
Clips land in bench/clips/ as 16 kHz mono wav, which is what whisper wants anyway.
"""

import base64
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

TTS = "http://127.0.0.1:8883/v1/audio/speech"
OUT = Path(__file__).parent / "clips"

VOICE_PT = "piper-pt-br-faber-medium-pt"
VOICE_EN = "kokoro-am-michael-en"

# Reference text per clip. Kept verbatim in REFERENCE.md so accuracy is checkable.
CLIPS = {
    "pt_03": (
        VOICE_PT,
        "pt",
        "Abre a porta da garagem e acende a luz da cozinha.",
    ),
    "pt_15": (
        VOICE_PT,
        "pt",
        "Preciso que voce transcreva o audio em portugues sem inventar palavras. "
        "O servidor roda numa placa de video e a latencia importa mais do que "
        "qualquer outra coisa neste momento.",
    ),
    "pt_60": (
        VOICE_PT,
        "pt",
        "Bom dia. Hoje eu quero medir quanto tempo o modelo leva para transcrever "
        "a minha voz, porque digitar comando por comando ja esta cansativo demais. "
        "A ideia e simples: eu falo, o computador escuta, e o texto aparece antes "
        "de eu terminar a frase. Se demorar tres segundos para uma frase de tres "
        "segundos, nao serve. Entao vamos testar o modelo grande na placa de video, "
        "vamos testar a versao turbo, e vamos testar o modelo medio tambem, para "
        "ver se a diferenca de qualidade justifica a diferenca de tempo. "
        "Palavras dificeis: manutencao, orcamento, exception, deployment, "
        "roboflow, kubernetes, transcricao simultanea. Se essas palavras sairem "
        "certas em portugues, o modelo passou no teste. Se sairem em ingles ou "
        "viraram outra coisa, o modelo reprovou, por mais rapido que ele seja.",
    ),
    "en_03": (
        VOICE_EN,
        "en",
        "Open the garage door and turn on the kitchen light.",
    ),
    "en_15": (
        VOICE_EN,
        "en",
        "I need you to transcribe this audio in English without making up any "
        "words. The server runs on a graphics card and right now latency matters "
        "more than anything else.",
    ),
    "en_60": (
        VOICE_EN,
        "en",
        "Good morning. Today I want to measure how long the model takes to "
        "transcribe my voice, because typing every command out by hand has become "
        "tiring. The idea is simple: I speak, the computer listens, and the text "
        "shows up before I finish the sentence. If it takes three seconds to "
        "handle a three second sentence, it is useless. So we are going to test "
        "the large model on the graphics card, we are going to test the turbo "
        "variant, and we are going to test the medium model too, to see whether "
        "the quality difference justifies the time difference. Hard words: "
        "maintenance, exception handling, deployment, roboflow, kubernetes, "
        "simultaneous transcription. If those come out right, the model passed. "
        "If they come out garbled, the model failed, however fast it was.",
    ),
}


def synth(voice: str, lang: str, text: str, dest: Path) -> None:
    body = json.dumps(
        {"input": text, "voice": voice, "lang": lang, "response_format": "wav"}
    ).encode()
    req = urllib.request.Request(
        TTS, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        payload = r.read()

    raw = dest.with_suffix(".raw")
    # The server answers either with wav bytes or with a JSON envelope carrying
    # base64 audio, depending on the backend that served the voice.
    if payload[:4] == b"RIFF":
        raw.write_bytes(payload)
    else:
        data = json.loads(payload)
        b64 = data.get("audio") or data.get("data") or data.get("b64_json")
        if not b64:
            raise SystemExit(f"unexpected TTS response: {list(data)[:8]}")
        raw.write_bytes(base64.b64decode(b64))

    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
         "-ar", "16000", "-ac", "1", str(dest)],
        check=True,
    )
    raw.unlink()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1:] or list(CLIPS)
    for name in only:
        voice, lang, text = CLIPS[name]
        dest = OUT / f"{name}.wav"
        print(f"synth {name} ({voice}) ...", flush=True)
        synth(voice, lang, text, dest)
        dur = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(dest)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        print(f"  -> {dest.name}  {float(dur):.2f}s")
