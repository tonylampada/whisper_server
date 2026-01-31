# Whisper Transcription Server

HTTP service for audio/video transcription using faster-whisper.

## Features

- **POST /transcribe** - Batch transcription of uploaded files
- **WebSocket /ws/transcribe** - Real-time streaming transcription
- **GET /** - Interactive test page with microphone recording
- **GET /health** - Health check endpoint

## Installation

```bash
cd ~/repos/whisper_server
uv sync
```

## Running

```bash
uv run python server.py
```

Or with uvicorn directly:

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8878
```

Server runs on http://localhost:8878

## API Usage

### Batch Transcription

```bash
curl -X POST http://localhost:8878/transcribe \
  -F "file=@audio.mp3" \
  -F "language=en"
```

Response:
```json
{
  "text": "Transcribed text here...",
  "language": "en",
  "language_probability": 0.98,
  "segments": [
    {"start": 0.0, "end": 2.5, "text": "Hello world"}
  ]
}
```

### Streaming Transcription (WebSocket)

Connect to `ws://localhost:8878/ws/transcribe` and send audio chunks as binary data. Server responds with JSON:

```json
{"text": "Chunk transcription", "language": "en"}
```

### Test Page

Open http://localhost:8878 in a browser to access the interactive test page with:
- File upload transcription
- Microphone recording
- Live streaming transcription

## Configuration

The server uses the `base` Whisper model by default. Edit `MODEL_NAME` in `server.py` to change:
- `tiny` - Fastest, lowest accuracy
- `base` - Good balance (default)
- `small` - Better accuracy
- `medium` - High accuracy
- `large-v3` - Best accuracy, slowest

## Supported Formats

Any format supported by faster-whisper/ffmpeg:
- Audio: wav, mp3, flac, ogg, webm, m4a
- Video: mp4, webm, mkv, avi, mov (audio track extracted)
