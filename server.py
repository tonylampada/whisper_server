#!/usr/bin/env python3
"""
Whisper transcription HTTP server.
- POST /transcribe - Batch transcription
- WebSocket /ws/transcribe - Streaming transcription
- GET / - Test page with microphone recording
"""

import os
import tempfile
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import cuda_libs  # noqa: F401  (must precede faster_whisper: preloads libcublas/libcudnn)
from faster_whisper import WhisperModel


# Global model (loaded once)
model: WhisperModel | None = None

# large-v3-turbo on the GPU: fastest configuration measured that still gets the
# Portuguese right (bench/RESULTS.md). It beats large-v3 on accuracy too, because
# large-v3 drops whole phrases on longer clips.
MODEL_NAME = os.environ.get("WHISPER_MODEL", "large-v3-turbo")


def pick_device() -> tuple[str, str]:
    """Return (device, compute_type): the GPU if CTranslate2 can reach it."""
    if os.environ.get("WHISPER_DEVICE") == "cpu":
        return "cpu", "int8"
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0 and cuda_libs.preload():
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def get_model() -> WhisperModel:
    """Get or load the Whisper model."""
    global model
    if model is None:
        device, compute_type = pick_device()
        print(f"Loading Whisper model '{MODEL_NAME}' on {device}/{compute_type}...")
        model = WhisperModel(MODEL_NAME, device=device, compute_type=compute_type)
        print("Model loaded!")
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load model on startup."""
    get_model()
    yield


app = FastAPI(
    title="Whisper Transcription Server",
    description="HTTP service for audio/video transcription using faster-whisper",
    version="1.0.0",
    lifespan=lifespan,
)


def transcribe_audio(audio_path: Path, language: str | None = None) -> dict:
    """
    Transcribe audio file and return result dict.
    """
    whisper = get_model()
    
    segments, info = whisper.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    # Collect segments
    text_parts = []
    segment_list = []
    for segment in segments:
        text_parts.append(segment.text.strip())
        segment_list.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })
    
    return {
        "text": " ".join(text_parts),
        "language": info.language,
        "language_probability": info.language_probability,
        "segments": segment_list
    }


@app.get("/", response_class=HTMLResponse)
async def test_page():
    """Test page with microphone recording."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Whisper Transcription Test</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }
        h1 { color: #00d9ff; }
        .section {
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
        h2 { color: #00d9ff; margin-top: 0; }
        button {
            background: #00d9ff;
            color: #1a1a2e;
            border: none;
            padding: 12px 24px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            margin: 5px;
        }
        button:hover { background: #00b8d9; }
        button:disabled { background: #555; cursor: not-allowed; }
        button.recording { background: #ff4757; }
        #result, #streamResult {
            background: #0f0f23;
            padding: 15px;
            border-radius: 5px;
            margin-top: 15px;
            min-height: 100px;
            white-space: pre-wrap;
            font-family: monospace;
        }
        .status { color: #ffd700; font-style: italic; }
        input[type="file"] { margin: 10px 0; }
        select {
            padding: 8px;
            border-radius: 5px;
            background: #0f0f23;
            color: #eee;
            border: 1px solid #00d9ff;
        }
        label { display: block; margin: 10px 0 5px; }
        .tabs { display: flex; gap: 10px; margin-bottom: 20px; }
        .tab {
            padding: 10px 20px;
            background: #16213e;
            border-radius: 5px 5px 0 0;
            cursor: pointer;
        }
        .tab.active { background: #00d9ff; color: #1a1a2e; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
    </style>
</head>
<body>
    <h1>🎤 Whisper Transcription Server</h1>
    
    <div class="tabs">
        <div class="tab active" onclick="switchTab('batch')">Batch Upload</div>
        <div class="tab" onclick="switchTab('stream')">Live Streaming</div>
    </div>
    
    <!-- Batch Transcription -->
    <div id="batchTab" class="tab-content active">
        <div class="section">
            <h2>📤 Upload Audio/Video</h2>
            <input type="file" id="fileInput" accept="audio/*,video/*">
            <label>Language (optional):
                <select id="language">
                    <option value="">Auto-detect</option>
                    <option value="en">English</option>
                    <option value="pt">Portuguese</option>
                    <option value="es">Spanish</option>
                    <option value="fr">French</option>
                    <option value="de">German</option>
                </select>
            </label>
            <button onclick="uploadFile()">Transcribe File</button>
            <div id="result"><span class="status">Upload a file to transcribe...</span></div>
        </div>
        
        <div class="section">
            <h2>🎙️ Record from Microphone</h2>
            <button id="recordBtn" onclick="toggleRecording()">Start Recording</button>
            <button onclick="transcribeRecording()" id="transcribeBtn" disabled>Transcribe Recording</button>
            <div id="recordStatus" class="status"></div>
        </div>
    </div>
    
    <!-- Streaming Transcription -->
    <div id="streamTab" class="tab-content">
        <div class="section">
            <h2>🔴 Live Streaming Transcription</h2>
            <p>Transcribes audio in real-time as you speak.</p>
            <button id="streamBtn" onclick="toggleStreaming()">Start Streaming</button>
            <div id="streamResult"><span class="status">Click Start Streaming to begin...</span></div>
        </div>
    </div>

    <script>
        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;
        let recordedBlob = null;
        let ws = null;
        let isStreaming = false;
        let streamRecorder = null;

        function switchTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            
            if (tab === 'batch') {
                document.querySelector('.tab:first-child').classList.add('active');
                document.getElementById('batchTab').classList.add('active');
            } else {
                document.querySelector('.tab:last-child').classList.add('active');
                document.getElementById('streamTab').classList.add('active');
            }
        }

        async function uploadFile() {
            const fileInput = document.getElementById('fileInput');
            const language = document.getElementById('language').value;
            const result = document.getElementById('result');
            
            if (!fileInput.files[0]) {
                result.innerHTML = '<span class="status">Please select a file first.</span>';
                return;
            }
            
            result.innerHTML = '<span class="status">Transcribing... please wait...</span>';
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            if (language) formData.append('language', language);
            
            try {
                const response = await fetch('/transcribe', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                
                if (data.error) {
                    result.innerHTML = `<span style="color: #ff4757;">Error: ${data.error}</span>`;
                } else {
                    result.innerHTML = `Language: ${data.language} (${(data.language_probability * 100).toFixed(1)}%)\n\n${data.text}`;
                }
            } catch (err) {
                result.innerHTML = `<span style="color: #ff4757;">Error: ${err.message}</span>`;
            }
        }

        async function toggleRecording() {
            const btn = document.getElementById('recordBtn');
            const status = document.getElementById('recordStatus');
            const transcribeBtn = document.getElementById('transcribeBtn');
            
            if (!isRecording) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];
                    
                    mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
                    mediaRecorder.onstop = () => {
                        recordedBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        transcribeBtn.disabled = false;
                        status.textContent = 'Recording saved. Click Transcribe Recording.';
                    };
                    
                    mediaRecorder.start();
                    isRecording = true;
                    btn.textContent = 'Stop Recording';
                    btn.classList.add('recording');
                    status.textContent = 'Recording...';
                    transcribeBtn.disabled = true;
                } catch (err) {
                    status.textContent = 'Error: ' + err.message;
                }
            } else {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(t => t.stop());
                isRecording = false;
                btn.textContent = 'Start Recording';
                btn.classList.remove('recording');
            }
        }

        async function transcribeRecording() {
            if (!recordedBlob) return;
            
            const result = document.getElementById('result');
            const language = document.getElementById('language').value;
            
            result.innerHTML = '<span class="status">Transcribing recording... please wait...</span>';
            
            const formData = new FormData();
            formData.append('file', recordedBlob, 'recording.webm');
            if (language) formData.append('language', language);
            
            try {
                const response = await fetch('/transcribe', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                
                if (data.error) {
                    result.innerHTML = `<span style="color: #ff4757;">Error: ${data.error}</span>`;
                } else {
                    result.innerHTML = `Language: ${data.language} (${(data.language_probability * 100).toFixed(1)}%)\n\n${data.text}`;
                }
            } catch (err) {
                result.innerHTML = `<span style="color: #ff4757;">Error: ${err.message}</span>`;
            }
        }

        async function toggleStreaming() {
            const btn = document.getElementById('streamBtn');
            const result = document.getElementById('streamResult');
            
            if (!isStreaming) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    
                    // Connect WebSocket
                    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
                    ws = new WebSocket(`${protocol}//${location.host}/ws/transcribe`);
                    
                    ws.onopen = () => {
                        result.innerHTML = '<span class="status">Connected. Speak now...</span>';
                        
                        // Start recording and sending chunks
                        streamRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
                        streamRecorder.ondataavailable = async (e) => {
                            if (ws.readyState === WebSocket.OPEN && e.data.size > 0) {
                                ws.send(e.data);
                            }
                        };
                        streamRecorder.start(1000); // Send chunk every 1 second
                    };
                    
                    ws.onmessage = (e) => {
                        const data = JSON.parse(e.data);
                        if (data.error) {
                            result.innerHTML += `\n<span style="color: #ff4757;">Error: ${data.error}</span>`;
                        } else {
                            // Server sends the transcript of everything so far,
                            // not just the newest chunk, so replace rather than append.
                            result.textContent = data.text;
                        }
                    };
                    
                    ws.onclose = () => {
                        result.innerHTML += '\n<span class="status">Disconnected.</span>';
                        stopStreaming();
                    };
                    
                    ws.onerror = (err) => {
                        result.innerHTML += `\n<span style="color: #ff4757;">WebSocket error</span>`;
                    };
                    
                    isStreaming = true;
                    btn.textContent = 'Stop Streaming';
                    btn.classList.add('recording');
                    
                } catch (err) {
                    result.innerHTML = `<span style="color: #ff4757;">Error: ${err.message}</span>`;
                }
            } else {
                stopStreaming();
            }
        }

        function stopStreaming() {
            const btn = document.getElementById('streamBtn');
            
            if (streamRecorder && streamRecorder.state !== 'inactive') {
                streamRecorder.stop();
                streamRecorder.stream.getTracks().forEach(t => t.stop());
            }
            if (ws) {
                ws.close();
                ws = null;
            }
            
            isStreaming = false;
            btn.textContent = 'Start Streaming';
            btn.classList.remove('recording');
        }
    </script>
</body>
</html>
"""


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(None)
):
    """
    Transcribe uploaded audio/video file.
    
    - **file**: Audio or video file (wav, mp3, webm, mp4, etc.)
    - **language**: Optional language code (en, pt, es, fr, de). Auto-detects if not provided.
    
    Returns transcription text, detected language, and segment timings.
    """
    # Save uploaded file to temp
    suffix = Path(file.filename or "audio.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)
    
    try:
        # Run transcription in thread pool to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            None, transcribe_audio, tmp_path, language
        )
        return result
    except Exception as e:
        return {"error": str(e)}
    finally:
        tmp_path.unlink(missing_ok=True)


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """
    Streaming transcription via WebSocket.

    Client sends audio chunks (binary), server replies with the transcript of
    everything sent so far, so partial text arrives while the speaker is still
    talking. Send the text frame "reset" to drop the buffer and start a new
    utterance.

    Chunks are accumulated rather than transcribed one by one: MediaRecorder
    emits the container header only in its first chunk, so every later chunk on
    its own is undecodable audio. Appending them back onto the header is what
    makes them playable again.
    """
    await websocket.accept()
    buffer = bytearray()

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("text") == "reset":
                buffer.clear()
                await websocket.send_json({"text": "", "reset": True})
                continue
            data = message.get("bytes")
            if not data:
                continue

            buffer.extend(data)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
                tmp.write(buffer)
                tmp_path = Path(tmp.name)

            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, transcribe_audio, tmp_path, None
                )
                await websocket.send_json({
                    "text": result["text"],
                    "language": result["language"],
                    "partial": True,
                })
            except Exception as e:
                await websocket.send_json({"error": str(e)})
            finally:
                tmp_path.unlink(missing_ok=True)

    except WebSocketDisconnect:
        pass


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model": MODEL_NAME}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8878)
