"""Make the pip-installed CUDA libraries visible to CTranslate2.

CTranslate2 dlopens libcublas/libcudnn by soname and does not know about the
`nvidia-*-cu12` wheels, which drop them inside site-packages instead of on the
loader path. Loading them RTLD_GLOBAL here puts them in the process before
CTranslate2 asks, so no LD_LIBRARY_PATH juggling is needed at launch.

Import this before faster_whisper. No-op when the wheels are absent (CPU-only).
"""

import ctypes
import sysconfig
from pathlib import Path

_SONAMES = [
    "cublas/lib/libcublas.so.12",
    "cublas/lib/libcublasLt.so.12",
    "cudnn/lib/libcudnn.so.9",
]


def preload() -> bool:
    """Return True if every CUDA library was loaded."""
    root = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    loaded = 0
    for rel in _SONAMES:
        lib = root / rel
        if not lib.exists():
            continue
        try:
            ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
            loaded += 1
        except OSError:
            pass
    return loaded == len(_SONAMES)


preload()
