
import numpy as np
import gzip, struct, urllib.request, os
from pathlib import Path

URL_BASES = [
    "http://yann.lecun.com/exdb/mnist/",
    # Fallback mirrors (kept in case the main host is slow/unavailable)
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}

def _download_file(fname: str, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / fname
    if dest.exists():
        return dest
    last_err = None
    for base in URL_BASES:
        url = base + fname
        try:
            urllib.request.urlretrieve(url, dest.as_posix())
            return dest
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"No se pudo descargar {fname}: {last_err}")

def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, 'rb') as f:
        _, num, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num, rows*cols).astype(np.float32) / 255.0

def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, 'rb') as f:
        _, num = struct.unpack('>II', f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(int)

def download_and_load_mnist(data_subdir: str = "mnist", cache_npz: bool = True):
    """
    Descarga si es necesario los ficheros IDX a data/<data_subdir>/ y devuelve (X, y).
    - X shape: (70000, 784) en float32 normalizado a [0,1]
    - y shape: (70000,) en int
    No usa sklearn.
    """
    base_dir = Path(__file__).resolve().parent / data_subdir
    base_dir.mkdir(parents=True, exist_ok=True)
    cache_path = base_dir / "mnist.npz"

    if cache_npz and cache_path.exists():
        cached = np.load(cache_path)
        return cached["X"], cached["y"]

    paths = {k: _download_file(v, base_dir) for k, v in FILES.items()}

    X_train = _read_idx_images(paths["train_images"])
    y_train = _read_idx_labels(paths["train_labels"])
    X_test  = _read_idx_images(paths["test_images"])
    y_test  = _read_idx_labels(paths["test_labels"])

    X = np.vstack([X_train, X_test]).astype(np.float32)
    y = np.concatenate([y_train, y_test]).astype(int)

    if cache_npz:
        np.savez_compressed(cache_path, X=X, y=y)

    return X, y
