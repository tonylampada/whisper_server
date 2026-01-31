#!/usr/bin/env python3
"""Entry point for whisper-server."""
from server import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8878)
