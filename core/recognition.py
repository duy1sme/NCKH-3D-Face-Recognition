import os
from typing import List, Optional

import cv2
import numpy as np
import onnxruntime as ort

_SESSION_CACHE: dict = {}


def _find_arcface_onnx() -> Optional[str]:
    env_path = os.environ.get("FACE3D_ARCFACE_ONNX", "").strip()
    if env_path and os.path.isfile(env_path):
        return os.path.abspath(env_path)

    home = os.path.expanduser("~")
    buffalo_dir = os.path.join(home, ".insightface", "models", "buffalo_l")
    candidates = [
        os.path.join(buffalo_dir, "w600k_r50.onnx"),
        os.path.join(buffalo_dir, "W600K_R50.ONNX"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    if os.path.isdir(buffalo_dir):
        for fn in os.listdir(buffalo_dir):
            if fn.lower() == "w600k_r50.onnx":
                p = os.path.join(buffalo_dir, fn)
                if os.path.isfile(p):
                    return p
    return None


def _providers() -> List[str]:
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def ensure_arcface_session() -> ort.InferenceSession:
    onnx_path = _find_arcface_onnx()
    if onnx_path is None:
        raise RuntimeError(
            "ArcFace ONNX not found. Run once to auto-download buffalo_l via insightface:\n"
            "  python -c \"from insightface.app import FaceAnalysis; "
            "app=FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); "
            "app.prepare(ctx_id=-1, det_size=(640,640))\""
        )

    providers = _providers()
    key = (onnx_path, tuple(providers))
    if key in _SESSION_CACHE:
        return _SESSION_CACHE[key]

    sess = ort.InferenceSession(onnx_path, providers=providers)
    _SESSION_CACHE[key] = sess
    return sess


def _select_embedding_output(outs, out_meta):
    candidates = []
    for meta, arr_raw in zip(out_meta, outs):
        arr = np.asarray(arr_raw)
        if arr.ndim == 0:
            continue
        batch = max(1, int(arr.shape[0])) if arr.ndim >= 1 else 1
        flat = int(arr.size // batch)
        name = (meta.name or "").lower()
        score = 0
        if ("fc1" in name) or ("embedding" in name) or ("feat" in name):
            score += 4
        if flat in (256, 384, 512, 1024):
            score += 3
        if 128 <= flat <= 2048:
            score += 1
        if batch >= 1:
            score += 1
        candidates.append((score, flat, arr))

    if not candidates:
        raise RuntimeError("ArcFace ONNX produced no valid output tensor.")

    candidates.sort(key=lambda x: (x[0], -abs(x[1] - 512)), reverse=True)
    best_score, best_flat, best_arr = candidates[0]
    if best_score <= 0 or not (128 <= best_flat <= 2048):
        raise RuntimeError("Cannot determine ArcFace embedding head.")
    return best_arr


def _to_bgr_112(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("Input image is None")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Invalid image shape: {img.shape}")

    if img.shape[:2] != (112, 112):
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def arcface_embed(sess: ort.InferenceSession, img_bgr_112: np.ndarray) -> np.ndarray:
    img = _to_bgr_112(img_bgr_112)
    blob = cv2.dnn.blobFromImage(
        img,
        scalefactor=1.0 / 127.5,
        size=(112, 112),
        mean=(127.5, 127.5, 127.5),
        swapRB=True,
        crop=False,
    ).astype(np.float32)

    input_name = sess.get_inputs()[0].name
    outs = sess.run(None, {input_name: blob})
    if not outs:
        raise RuntimeError("ArcFace ONNX produced no outputs.")

    best_arr = _select_embedding_output(outs, sess.get_outputs())
    feat = np.asarray(best_arr, dtype=np.float32).reshape(best_arr.shape[0], -1)[0]
    if feat.size == 0 or not np.all(np.isfinite(feat)):
        raise RuntimeError("ArcFace embedding invalid.")

    norm = float(np.linalg.norm(feat))
    if norm < 1e-12:
        raise RuntimeError("ArcFace embedding norm too small.")
    return feat / norm


def arcface_embed_batch(sess: ort.InferenceSession, imgs_bgr_112) -> np.ndarray:
    if imgs_bgr_112 is None:
        return np.zeros((0, 1), dtype=np.float32)
    if isinstance(imgs_bgr_112, np.ndarray) and imgs_bgr_112.ndim == 4:
        imgs = [imgs_bgr_112[i] for i in range(imgs_bgr_112.shape[0])]
    else:
        imgs = list(imgs_bgr_112)
    if not imgs:
        return np.zeros((0, 1), dtype=np.float32)

    processed = [_to_bgr_112(img) for img in imgs]
    blob = cv2.dnn.blobFromImages(
        processed,
        scalefactor=1.0 / 127.5,
        size=(112, 112),
        mean=(127.5, 127.5, 127.5),
        swapRB=True,
        crop=False,
    ).astype(np.float32)

    input_name = sess.get_inputs()[0].name
    outs = sess.run(None, {input_name: blob})
    if not outs:
        raise RuntimeError("ArcFace ONNX produced no outputs.")

    best_arr = _select_embedding_output(outs, sess.get_outputs())
    feat = np.asarray(best_arr, dtype=np.float32).reshape(best_arr.shape[0], -1)
    if feat.size == 0 or not np.all(np.isfinite(feat)):
        raise RuntimeError("ArcFace embedding invalid.")

    norms = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-12
    return feat / norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

