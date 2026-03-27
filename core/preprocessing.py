import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import onnxruntime as ort
try:
    from insightface.utils import face_align
except Exception:
    face_align = None

_FACE_APP_CACHE = {}

_ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _normalize_output_size(output_size) -> tuple[int, int]:
    if isinstance(output_size, int):
        if output_size <= 0:
            raise ValueError("output_size must be positive")
        return int(output_size), int(output_size)

    if not hasattr(output_size, "__len__") or len(output_size) != 2:
        raise ValueError("output_size must be int or tuple/list of 2 ints")

    out_w = int(output_size[0])
    out_h = int(output_size[1])
    if out_w <= 0 or out_h <= 0:
        raise ValueError("output_size values must be positive")
    return out_w, out_h


def _resolve_border_mode(border_mode) -> int:
    if isinstance(border_mode, int):
        return int(border_mode)

    token = str(border_mode).strip().lower()
    if token in {"replicate", "edge"}:
        return cv2.BORDER_REPLICATE
    if token in {"reflect", "reflect101", "reflect_101"}:
        return cv2.BORDER_REFLECT_101
    return cv2.BORDER_CONSTANT


def _parse_det_size(det_size_env: str | None) -> tuple[int, int]:
    if not det_size_env:
        return (640, 640)

    token = str(det_size_env).strip().lower().replace(" ", "")
    if not token:
        return (640, 640)

    if "x" in token:
        a, b = token.split("x", 1)
        try:
            w = int(a)
            h = int(b)
        except Exception:
            return (640, 640)
        if w > 0 and h > 0:
            return (w, h)
        return (640, 640)

    try:
        side = int(token)
    except Exception:
        return (640, 640)
    if side <= 0:
        return (640, 640)
    return (side, side)


def _runtime_det_size() -> tuple[int, int]:
    return _parse_det_size(os.environ.get("FACE_DET_SIZE"))


def _get_face_app(det_size=(640, 640)):
    key = (int(det_size[0]), int(det_size[1]))
    if key in _FACE_APP_CACHE:
        return _FACE_APP_CACHE[key]

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = 0
    else:
        providers = ["CPUExecutionProvider"]
        ctx_id = -1

    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=key)

    _FACE_APP_CACHE[key] = app
    return _FACE_APP_CACHE[key]


def _load_bgr_image(image_source):
    if isinstance(image_source, str):
        img_bgr = cv2.imread(image_source)
        if img_bgr is None:
            raise ValueError(f"Image load failed: {image_source}")
        return img_bgr

    if not isinstance(image_source, np.ndarray):
        raise TypeError("image_source must be a file path or numpy array")

    img = image_source
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Invalid image shape: {img.shape}")

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def _center_crop_resize_rgb(img_bgr, out_w: int, out_h: int):
    h, w = img_bgr.shape[:2]
    side = min(h, w)
    y0 = max(0, (h - side) // 2)
    x0 = max(0, (w - side) // 2)
    crop = img_bgr[y0 : y0 + side, x0 : x0 + side]
    interpolation = cv2.INTER_AREA if side >= max(out_w, out_h) else cv2.INTER_CUBIC
    crop = cv2.resize(crop, (out_w, out_h), interpolation=interpolation)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def process_face_retina(
    image_source,
    output_size=112,
    min_face_size: int = 20,
    allow_fallback: bool = True,
    template_scale: float = 1.0,
    border_mode="constant",
):
    img_bgr = _load_bgr_image(image_source)
    out_w, out_h = _normalize_output_size(output_size)
    template_scale = float(template_scale)
    if not np.isfinite(template_scale) or template_scale <= 0:
        template_scale = 1.0
    template_scale = float(np.clip(template_scale, 0.75, 1.60))
    border_mode_cv = _resolve_border_mode(border_mode)

    if img_bgr.shape[0] < 40 or img_bgr.shape[1] < 40:
        raise ValueError("Image too small")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    app = _get_face_app(det_size=_runtime_det_size())

    detect_attempts = [(img_bgr, 1.0)]
    h, w = img_bgr.shape[:2]
    short_side = float(max(1, min(h, w)))
    for target_short in (220.0, 320.0, 420.0):
        scale = min(4.0, target_short / short_side)
        if scale <= 1.05:
            continue
        if any(abs(scale - s) < 0.08 for _, s in detect_attempts):
            continue
        up = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        detect_attempts.append((up, scale))

    faces = []
    used_scale = 1.0
    for det_img, det_scale in detect_attempts:
        probe_images = [det_img]
        probe_images.append(cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB))
        gray = cv2.cvtColor(det_img, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        probe_images.append(cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR))

        for probe in probe_images:
            try:
                cur_faces = app.get(probe)  # InsightFace detector thường dùng BGR (cv2 format)
            except Exception:
                cur_faces = []
            if len(cur_faces) > 0:
                faces = cur_faces
                used_scale = det_scale
                break
        if len(faces) > 0:
            break

    if len(faces) == 0:
        if allow_fallback:
            return _center_crop_resize_rgb(img_bgr, out_w, out_h)
        raise ValueError("No face detected (set FACE_DET_FALLBACK=1 to enable center-crop fallback)")

    face = max(
        faces,
        key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
    )

    bbox = np.asarray(face.bbox, dtype=np.float32) / float(used_scale)
    x1, y1, x2, y2 = bbox.tolist()
    if (x2 - x1) < min_face_size or (y2 - y1) < min_face_size:
        if allow_fallback:
            return _center_crop_resize_rgb(img_bgr, out_w, out_h)
        raise ValueError("Face too small")

    src = np.asarray(face.kps, dtype=np.float32) / float(used_scale)
    if src.shape != (5, 2):
        if allow_fallback:
            return _center_crop_resize_rgb(img_bgr, out_w, out_h)
        raise ValueError(f"Unexpected landmark shape: {src.shape}")

    if (
        face_align is not None
        and out_w == 112
        and out_h == 112
        and abs(template_scale - 1.0) <= 1e-6
        and border_mode_cv == cv2.BORDER_CONSTANT
    ):
        aligned_bgr = face_align.norm_crop(img_bgr, landmark=src, image_size=112)
        return cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)

    dst = _ARCFACE_TEMPLATE_112.copy()
    if abs(template_scale - 1.0) > 1e-6:
        center = np.mean(dst, axis=0, keepdims=True)
        dst = (dst - center) * template_scale + center
    dst[:, 0] *= out_w / 112.0
    dst[:, 1] *= out_h / 112.0

    affine, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if affine is None:
        raise RuntimeError("Failed to estimate affine transform for face alignment")

    aligned_rgb = cv2.warpAffine(
        img_rgb,
        affine,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode_cv,
        borderValue=(0, 0, 0),
    )
    return aligned_rgb
