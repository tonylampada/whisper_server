"""Tests for the two seams the latency work introduced.

Both stub the model out: what is worth holding here is the device choice and the
websocket's chunk accumulation, not whisper's output.
"""

import server
from fastapi.testclient import TestClient


def test_pick_device_honours_cpu_override(monkeypatch):
    monkeypatch.setenv("WHISPER_DEVICE", "cpu")
    assert server.pick_device() == ("cpu", "int8")


def test_pick_device_falls_back_to_cpu_without_cuda(monkeypatch):
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    monkeypatch.setattr(server.cuda_libs, "preload", lambda: False)
    assert server.pick_device() == ("cpu", "int8")


def test_websocket_accumulates_chunks(monkeypatch):
    """Every chunk after the first is a headerless fragment, so the server must
    transcribe the whole buffer rather than the chunk it just received."""
    seen = []

    def fake_transcribe(path, language=None):
        seen.append(path.read_bytes())
        return {"text": f"len={len(seen[-1])}", "language": "pt", "segments": []}

    monkeypatch.setattr(server, "transcribe_audio", fake_transcribe)

    with TestClient(server.app).websocket_connect("/ws/transcribe") as ws:
        ws.send_bytes(b"HEADER")
        assert ws.receive_json()["text"] == "len=6"
        ws.send_bytes(b"FRAG")
        assert ws.receive_json()["text"] == "len=10"

        ws.send_text("reset")
        assert ws.receive_json()["reset"] is True
        ws.send_bytes(b"NEW")
        assert ws.receive_json()["text"] == "len=3"

    assert seen[1].startswith(b"HEADER")


def test_transcribe_passes_prompt_and_hotwords(monkeypatch):
    """The optional bias fields reach the model; absent, they arrive as None."""
    calls = []

    def fake_transcribe(path, language=None, prompt=None, hotwords=None):
        calls.append((language, prompt, hotwords))
        return {"text": "ok", "language": "pt", "segments": []}

    monkeypatch.setattr(server, "transcribe_audio", fake_transcribe)
    client = TestClient(server.app)
    files = {"file": ("a.wav", b"RIFF", "audio/wav")}
    r = client.post("/transcribe", files=files, data={"language": "pt", "prompt": "Câmbio.", "hotwords": "câmbio"})
    assert r.json()["text"] == "ok"
    r = client.post("/transcribe", files=files, data={"language": "pt"})
    assert calls == [("pt", "Câmbio.", "câmbio"), ("pt", None, None)]


def test_transcribe_audio_forwards_bias_to_whisper(monkeypatch):
    """Empty strings are no bias at all: None, not an empty prompt."""
    seen = {}

    class Info:
        language = "pt"
        language_probability = 1.0

    class FakeModel:
        def transcribe(self, path, **kw):
            seen.update(kw)
            return iter(()), Info()

    monkeypatch.setattr(server, "get_model", lambda: FakeModel())
    server.transcribe_audio(server.Path("x.wav"), "pt", "Câmbio.", "câmbio")
    assert (seen["initial_prompt"], seen["hotwords"]) == ("Câmbio.", "câmbio")
    server.transcribe_audio(server.Path("x.wav"), "pt", "", "")
    assert (seen["initial_prompt"], seen["hotwords"]) == (None, None)
