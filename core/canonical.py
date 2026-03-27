import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import trimesh

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(ROOT_DIR)

if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore
if not hasattr(np, "int"):
    np.int = int  # type: ignore
if not hasattr(np, "float"):
    np.float = float  # type: ignore
if not hasattr(np, "complex"):
    np.complex = complex  # type: ignore
if not hasattr(np, "object"):
    np.object = object  # type: ignore
if not hasattr(np, "unicode"):
    np.unicode = str  # type: ignore
if not hasattr(np, "str"):
    np.str = str  # type: ignore

ImageInput = Union[str, np.ndarray]

_DECA_CACHE: Dict[str, object] = {}
_DECA_CFG_CACHE: Dict[str, object] = {}
_DETAIL_DIM_FALLBACK: Optional[int] = None


def _default_model_tar() -> Optional[str]:
    candidates = [
        os.path.join(PROJECT_DIR, "models", "team_face3d_model.tar"),
        os.path.join(PROJECT_DIR, "models", "model.tar"),
        os.path.join(PROJECT_DIR, "models", "deca_model.tar"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def _resolve_deca_repo() -> str:
    env_path = (
        os.environ.get("FACE3D_BACKBONE_PATH", "").strip()
        or os.environ.get("DECA_REPO_PATH", "").strip()
    )
    if env_path and os.path.isdir(env_path):
        return env_path
    candidates = [
        os.path.join(PROJECT_DIR, "face3d-backbone"),
        os.path.join(PROJECT_DIR, "DECA-master"),
        os.path.join(PROJECT_DIR, "DECA"),
        os.path.join(PROJECT_DIR, "deca"),
        os.path.join(os.path.dirname(PROJECT_DIR), "face3d-backbone"),
        os.path.join(os.path.dirname(PROJECT_DIR), "DECA-master"),
        os.path.join(os.path.dirname(PROJECT_DIR), "DECA"),
        os.path.join(os.getcwd(), "face3d-backbone"),
        os.path.join(os.getcwd(), "DECA-master"),
        os.path.join(os.getcwd(), "DECA"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return ""


def _import_deca_modules():
    deca_repo = _resolve_deca_repo()
    if deca_repo and deca_repo not in sys.path:
        sys.path.insert(0, deca_repo)
    try:
        from decalib.deca import DECA  # type: ignore
        from decalib.utils.config import cfg as deca_cfg  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Khong import duoc backend DECA-based. "
            "Set FACE3D_BACKBONE_PATH (hoac DECA_REPO_PATH) toi thu muc DECA-master."
        ) from exc
    return DECA, deca_cfg


def _get_deca(device: torch.device, model_tar: Optional[str]) -> tuple:
    if not model_tar:
        model_tar = (
            os.environ.get("FACE3D_MODEL_TAR", "").strip()
            or os.environ.get("DECA_MODEL_TAR", "").strip()
            or None
        )
    if not model_tar:
        model_tar = _default_model_tar()

    key = f"{device}|{model_tar or ''}"
    if key in _DECA_CACHE:
        return _DECA_CACHE[key], _DECA_CFG_CACHE[key]

    DECA, deca_cfg = _import_deca_modules()
    if model_tar:
        deca_cfg.pretrained_modelpath = os.path.abspath(model_tar)
    try:
        if hasattr(deca_cfg, "model"):
            if hasattr(deca_cfg.model, "use_tex"):
                deca_cfg.model.use_tex = True
            if hasattr(deca_cfg.model, "extract_tex"):
                deca_cfg.model.extract_tex = False
    except Exception:
        pass
    try:
        deca_cfg.rasterizer_type = "pytorch3d"
    except Exception:
        pass

    model = DECA(config=deca_cfg, device=device)
    _DECA_CACHE[key] = model
    _DECA_CFG_CACHE[key] = deca_cfg
    return model, deca_cfg


def _load_rgb_image(image_input: ImageInput, input_is_rgb: bool = False) -> np.ndarray:
    if isinstance(image_input, str):
        img_bgr = cv2.imread(image_input)
        if img_bgr is None:
            raise FileNotFoundError(f"Khong tim thay anh: {image_input}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if not isinstance(image_input, np.ndarray):
        raise TypeError("img_path must be a file path or numpy array")
    img = image_input
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB if not input_is_rgb else cv2.COLOR_RGBA2RGB)
    elif img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Invalid image shape: {img.shape}")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if input_is_rgb:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _prepare_tensor_224(img_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    img = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    arr = img.astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def _extract_faces(deca_model) -> np.ndarray:
    faces = None
    try:
        faces = getattr(getattr(deca_model, "render", None), "faces", None)
    except Exception:
        faces = None
    if faces is None:
        return np.zeros((0, 3), dtype=np.int64)
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu()
    arr = np.asarray(faces)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        return np.zeros((0, 3), dtype=np.int64)
    if arr.shape[1] != 3 and arr.shape[0] == 3:
        arr = arr.T
    arr = arr.astype(np.int64, copy=False)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    if int(arr.min()) >= 1:
        arr = arr - 1
    return np.ascontiguousarray(arr)


def _clone_codedict(codedict: dict) -> dict:
    out = {}
    for k, v in codedict.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.clone()
        else:
            out[k] = v
    return out


def _infer_detail_dim(codedict: dict) -> int:
    global _DETAIL_DIM_FALLBACK
    detail_t = codedict.get("detail", None) if isinstance(codedict, dict) else None
    if isinstance(detail_t, torch.Tensor):
        return int(detail_t.shape[-1])
    if _DETAIL_DIM_FALLBACK is not None:
        return int(_DETAIL_DIM_FALLBACK)
    env = os.environ.get("FACE3D_DETAIL_DIM", "").strip() or os.environ.get("DECA_DETAIL_DIM", "").strip()
    if env:
        try:
            _DETAIL_DIM_FALLBACK = int(env)
            return int(_DETAIL_DIM_FALLBACK)
        except Exception:
            _DETAIL_DIM_FALLBACK = 0
            return 0
    _DETAIL_DIM_FALLBACK = 0
    return 0


def _neutralize_codedict(
    codedict: dict,
    neutralize_exp: bool = True,
    neutralize_pose: bool = True,
    recenter_cam: bool = True,
) -> dict:
    if not isinstance(codedict, dict):
        return codedict
    out = _clone_codedict(codedict)
    if neutralize_exp and "exp" in out and isinstance(out["exp"], torch.Tensor):
        out["exp"] = torch.zeros_like(out["exp"])
    if neutralize_pose and "pose" in out and isinstance(out["pose"], torch.Tensor):
        out["pose"] = torch.zeros_like(out["pose"])
    if recenter_cam and "cam" in out and isinstance(out["cam"], torch.Tensor):
        cam = out["cam"].clone()
        if cam.ndim == 2 and cam.shape[1] >= 3:
            cam[:, 1:] = 0.0
            cam[:, 0] = torch.clamp(cam[:, 0], min=4.5, max=8.0)
            out["cam"] = cam
    return out


def _tensor_to_bgr_image(t: torch.Tensor) -> Optional[np.ndarray]:
    if t is None:
        return None
    x = t.detach().cpu().float()
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)
    arr = x.numpy()
    if arr.ndim != 3:
        return None
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    vmin, vmax = float(np.min(arr)), float(np.max(arr))
    if vmax <= 1.0 and vmin >= -1.0:
        arr = (arr + (0.0 if vmin >= 0 else 1.0)) / (1.0 if vmin >= 0 else 2.0)
    arr = np.clip(arr, 0.0, 1.0)
    return cv2.cvtColor((arr * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _tensor_to_bgr_image_at(t: torch.Tensor, idx: int) -> Optional[np.ndarray]:
    if t is None:
        return None
    x = t.detach().cpu().float()
    if x.ndim == 4:
        if idx < 0 or idx >= x.shape[0]:
            return None
        x = x[idx]
    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)
    arr = x.numpy()
    if arr.ndim != 3:
        return None
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    vmin, vmax = float(np.min(arr)), float(np.max(arr))
    if vmax <= 1.0 and vmin >= -1.0:
        arr = (arr + (0.0 if vmin >= 0 else 1.0)) / (1.0 if vmin >= 0 else 2.0)
    arr = np.clip(arr, 0.0, 1.0)
    return cv2.cvtColor((arr * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _score_frontal_candidate(bgr: np.ndarray) -> float:
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        return -1.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h < 4 or w < 4:
        return -1.0
    valid = float(np.mean(gray > 8))
    top = float(np.mean(gray[: h // 2, :] > 8))
    bottom = float(np.mean(gray[h // 2 :, :] > 8))
    balance = 1.0 - abs(top - bottom)
    center = gray[h // 4 : (3 * h) // 4, w // 4 : (3 * w) // 4]
    center_valid = float(np.mean(center > 8))
    return valid * 0.5 + balance * 0.3 + center_valid * 0.2


def _extract_candidate_bgr(val, idx: Optional[int] = None) -> Optional[np.ndarray]:
    if isinstance(val, torch.Tensor):
        if idx is None:
            return _tensor_to_bgr_image(val)
        return _tensor_to_bgr_image_at(val, idx)
    arr = np.asarray(val)
    if arr.ndim == 4:
        pick = 0 if idx is None else idx
        if pick < 0 or pick >= arr.shape[0]:
            return None
        arr = arr[pick]
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        return None
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    if arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _extract_frontal_bgr(opdict: dict, visdict: dict, img_rgb: np.ndarray) -> np.ndarray:
    candidates = []
    for src in (visdict, opdict):
        if not isinstance(src, dict):
            continue
        # Prefer natural-color outputs only. Exclude normal maps.
        for key in ("predicted_images", "rendered_images", "shape_images"):
            if key in src:
                bgr = _extract_candidate_bgr(src[key], idx=None)
                if bgr is not None:
                    candidates.append(bgr)
    if candidates:
        scored = sorted(
            ((float(_score_frontal_candidate(c)), c) for c in candidates),
            key=lambda x: x[0],
            reverse=True,
        )
        if scored[0][0] > 0.05:
            return scored[0][1]
    return cv2.cvtColor(cv2.resize(img_rgb, (224, 224)), cv2.COLOR_RGB2BGR)


def _extract_frontal_bgr_at(opdict: dict, visdict: dict, img_rgb: np.ndarray, idx: int) -> np.ndarray:
    candidates = []
    for src in (visdict, opdict):
        if not isinstance(src, dict):
            continue
        # Prefer natural-color outputs only. Exclude normal maps.
        for key in ("predicted_images", "rendered_images", "shape_images"):
            if key in src:
                bgr = _extract_candidate_bgr(src[key], idx=idx)
                if bgr is not None:
                    candidates.append(bgr)
    if candidates:
        scored = sorted(
            ((float(_score_frontal_candidate(c)), c) for c in candidates),
            key=lambda x: x[0],
            reverse=True,
        )
        if scored[0][0] > 0.05:
            return scored[0][1]
    return cv2.cvtColor(cv2.resize(img_rgb, (224, 224)), cv2.COLOR_RGB2BGR)


def _sample_vertex_colors(vertices: np.ndarray, img_rgb_224: np.ndarray) -> np.ndarray:
    x, y = vertices[:, 0], vertices[:, 1]
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    if (xmax - xmin) < 1e-8 or (ymax - ymin) < 1e-8:
        base = np.full((vertices.shape[0], 3), 180, dtype=np.uint8)
    else:
        px = np.clip(((x - xmin) / (xmax - xmin) * 223.0).round().astype(np.int32), 0, 223)
        py = np.clip((223.0 - (y - ymin) / (ymax - ymin) * 223.0).round().astype(np.int32), 0, 223)
        base = img_rgb_224[py, px, :]
    alpha = 255 * np.ones((base.shape[0], 1), dtype=np.uint8)
    return np.concatenate([base, alpha], axis=1)


def _extract_shape_feature(
    codedict: dict,
    coeff_clip: Optional[float] = 3.0,
    shape_scale: float = 1.0,
    use_detail: bool = True,
) -> np.ndarray:
    if not isinstance(codedict, dict):
        return np.zeros((100,), dtype=np.float32)

    alpha_t = codedict.get("shape", None)
    if alpha_t is None:
        return np.zeros((100,), dtype=np.float32)

    if isinstance(alpha_t, torch.Tensor):
        alpha = alpha_t.detach().cpu().numpy().astype(np.float32)
    else:
        alpha = np.asarray(alpha_t, dtype=np.float32)

    if alpha.ndim > 1:
        alpha = alpha[0]
    alpha = alpha.reshape(-1)

    if coeff_clip is not None and float(coeff_clip) > 0:
        alpha = np.clip(alpha, -float(coeff_clip), float(coeff_clip))
    alpha = alpha * float(shape_scale)

    parts = [alpha]

    if use_detail:
        detail_t = codedict.get("detail", None)
        if detail_t is not None:
            if isinstance(detail_t, torch.Tensor):
                detail = detail_t.detach().cpu().numpy().astype(np.float32)
            else:
                detail = np.asarray(detail_t, dtype=np.float32)
            if detail.ndim > 1:
                detail = detail[0]
            detail = detail.reshape(-1)
            if coeff_clip is not None and float(coeff_clip) > 0:
                detail = np.clip(detail, -float(coeff_clip), float(coeff_clip))
                parts.append(detail * 0.3)
        else:
            detail_dim = _infer_detail_dim(codedict)
            if detail_dim > 0:
                parts.append(np.zeros((detail_dim,), dtype=np.float32))

    feat = np.concatenate(parts, axis=0).astype(np.float32)
    norm = float(np.linalg.norm(feat))
    if norm > 1e-12:
        feat = feat / norm
    return feat


def _extract_shape_feature_at(
    codedict: dict,
    idx: int,
    coeff_clip: Optional[float] = 3.0,
    shape_scale: float = 1.0,
    use_detail: bool = True,
) -> np.ndarray:
    if not isinstance(codedict, dict):
        return np.zeros((100,), dtype=np.float32)

    alpha_t = codedict.get("shape", None)
    if alpha_t is None:
        return np.zeros((100,), dtype=np.float32)

    if isinstance(alpha_t, torch.Tensor):
        alpha_all = alpha_t.detach().cpu().numpy().astype(np.float32)
    else:
        alpha_all = np.asarray(alpha_t, dtype=np.float32)

    if alpha_all.ndim == 1:
        alpha = alpha_all.reshape(-1)
    else:
        if idx >= alpha_all.shape[0]:
            return np.zeros((alpha_all.shape[-1],), dtype=np.float32)
        alpha = alpha_all[idx].reshape(-1)

    if coeff_clip is not None and float(coeff_clip) > 0:
        alpha = np.clip(alpha, -float(coeff_clip), float(coeff_clip))
    alpha = alpha * float(shape_scale)

    parts = [alpha]

    if use_detail:
        detail_t = codedict.get("detail", None)
        if detail_t is not None:
            if isinstance(detail_t, torch.Tensor):
                detail_all = detail_t.detach().cpu().numpy().astype(np.float32)
            else:
                detail_all = np.asarray(detail_t, dtype=np.float32)
            if detail_all.ndim == 1:
                detail = detail_all.reshape(-1)
            else:
                if idx < detail_all.shape[0]:
                    detail = detail_all[idx].reshape(-1)
                else:
                    detail = None
            if detail is not None:
                if coeff_clip is not None and float(coeff_clip) > 0:
                    detail = np.clip(detail, -float(coeff_clip), float(coeff_clip))
                parts.append(detail * 0.3)
        else:
            detail_dim = _infer_detail_dim(codedict)
            if detail_dim > 0:
                parts.append(np.zeros((detail_dim,), dtype=np.float32))

    feat = np.concatenate(parts, axis=0).astype(np.float32)
    norm = float(np.linalg.norm(feat))
    if norm > 1e-12:
        feat = feat / norm
    return feat


_FLAME_IDENTITY_REGION_INDICES: Optional[np.ndarray] = None

def _get_identity_region_mask(n_verts: int) -> np.ndarray:
    if n_verts != 5023:
        return np.arange(n_verts)

    stable_mask = np.concatenate([
        np.arange(0, 1500),      # Vùng trán + mũi
        np.arange(1500, 2500),   # Vùng má + mắt
        np.arange(2500, 3200),   # Vùng miệng + cằm
    ])
    return stable_mask[stable_mask < n_verts]


def compute_vertex_shape_descriptor(
    vertices: np.ndarray,
    n_bins: int = 32,
    use_pairwise: bool = False,
) -> np.ndarray:
    if vertices is None or len(vertices) == 0:
        return np.zeros((n_bins * 3,), dtype=np.float32)

    verts = vertices.astype(np.float32)

    mask = _get_identity_region_mask(len(verts))
    verts = verts[mask]

    center = verts.mean(axis=0)
    verts = verts - center
    scale = np.linalg.norm(verts, axis=1).max()
    if scale > 1e-8:
        verts = verts / scale

    parts = []

    hist_z, _ = np.histogram(verts[:, 2], bins=n_bins, range=(-1.5, 1.5), density=True)
    parts.append(hist_z.astype(np.float32))

    hist_x, _ = np.histogram(verts[:, 0], bins=n_bins, range=(-1.5, 1.5), density=True)
    parts.append(hist_x.astype(np.float32))

    hist_y, _ = np.histogram(verts[:, 1], bins=n_bins, range=(-1.5, 1.5), density=True)
    parts.append(hist_y.astype(np.float32))

    if use_pairwise and len(verts) > 10:
        step = max(1, len(verts) // 200)
        sub = verts[::step][:200]
        dists = []
        for i in range(len(sub)):
            d = np.linalg.norm(sub[i+1:] - sub[i], axis=1)
            dists.extend(d.tolist())
        dists = np.array(dists, dtype=np.float32)
        hist_d, _ = np.histogram(dists, bins=n_bins, range=(0.0, 3.0), density=True)
        parts.append(hist_d.astype(np.float32))

    feat = np.concatenate(parts).astype(np.float32)
    norm = np.linalg.norm(feat)
    if norm > 1e-12:
        feat = feat / norm
    return feat


def calibrate_scores_zscore(
    scores: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    mu = float(np.mean(scores))
    sigma = float(np.std(scores)) + eps
    return (scores - mu) / sigma


def calibrate_scores_minmax(
    scores: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    lo, hi = float(np.min(scores)), float(np.max(scores))
    return (scores - lo) / (hi - lo + eps)


def fuse_scores_calibrated(
    scores_2d: np.ndarray,
    scores_3d: np.ndarray,
    weight_2d: float = 0.7,
    weight_3d: float = 0.3,
    method: str = "zscore",
) -> np.ndarray:
    s2 = np.asarray(scores_2d, dtype=np.float32)
    s3 = np.asarray(scores_3d, dtype=np.float32)

    if method == "zscore":
        s2_cal = calibrate_scores_zscore(s2)
        s3_cal = calibrate_scores_zscore(s3)
    elif method == "minmax":
        s2_cal = calibrate_scores_minmax(s2)
        s3_cal = calibrate_scores_minmax(s3)
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    total_w = weight_2d + weight_3d
    fused = (s2_cal * weight_2d + s3_cal * weight_3d) / total_w
    return fused.astype(np.float32)


def extract_coeffs_batch(
    imgs_rgb_224: list,
    device: torch.device = torch.device("cpu"),
    model_tar: Optional[str] = None,
    coeff_clip: Optional[float] = 3.0,
    shape_scale: float = 1.0,
    expression_scale: float = 1.0,  # Kept for API compat nhưng KHÔNG còn dùng
    use_detail: bool = True,
) -> list:
    if not imgs_rgb_224:
        return []

    if expression_scale != 1.0:
        import warnings
        warnings.warn(
            "expression_scale is deprecated in extract_coeffs_batch. "
            "Expression codes are no longer used for identity features. "
            "Use extract_coeffs_with_expression_batch() if you need them.",
            DeprecationWarning,
            stacklevel=2,
        )

    deca_model, _ = _get_deca(device=device, model_tar=model_tar)

    arr = []
    for img in imgs_rgb_224:
        if img.shape[:2] != (224, 224):
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        arr.append(img.astype(np.float32) / 255.0)
    ten = torch.from_numpy(np.stack(arr, axis=0)).permute(0, 3, 1, 2).to(device)

    with torch.no_grad():
        codedict = deca_model.encode(ten)

    outputs = []
    if isinstance(codedict, dict):
        n = len(imgs_rgb_224)
        for i in range(n):
            feat = _extract_shape_feature_at(
                codedict, idx=i,
                coeff_clip=coeff_clip,
                shape_scale=shape_scale,
                use_detail=use_detail,
            )
            outputs.append(feat)
    else:
        outputs = [np.zeros((100,), dtype=np.float32) for _ in imgs_rgb_224]

    return outputs


def extract_shape_only_batch(
    imgs_rgb_224: list,
    device: torch.device = torch.device("cpu"),
    model_tar: Optional[str] = None,
    coeff_clip: float = 3.0,
    return_vertex_descriptor: bool = False,
    shape_scale: float = 1.0,
    neutralize_exp: bool = True,
    neutralize_pose: bool = True,
) -> List[Dict[str, np.ndarray]]:
    if not imgs_rgb_224:
        return []

    deca_model, _ = _get_deca(device=device, model_tar=model_tar)

    arr = []
    for img in imgs_rgb_224:
        if img.shape[:2] != (224, 224):
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        arr.append(img.astype(np.float32) / 255.0)
    ten = torch.from_numpy(np.stack(arr, axis=0)).permute(0, 3, 1, 2).to(device)

    with torch.no_grad():
        codedict = deca_model.encode(ten)
        codedict_render = _neutralize_codedict(
            codedict, neutralize_exp=neutralize_exp, neutralize_pose=neutralize_pose
        )
        decoded = deca_model.decode(codedict_render)

    if isinstance(decoded, tuple):
        opdict = decoded[0] if len(decoded) > 0 else {}
    elif isinstance(decoded, dict):
        opdict = decoded
    else:
        opdict = {}

    outputs = []
    for i in range(len(imgs_rgb_224)):
        result = {}

        result['shape_feat'] = _extract_shape_feature_at(
            codedict, idx=i, coeff_clip=coeff_clip, shape_scale=shape_scale, use_detail=True
        )

        alpha_t = codedict.get("shape", None) if isinstance(codedict, dict) else None
        if alpha_t is not None:
            a = alpha_t.detach().cpu().numpy().astype(np.float32)
            result['alpha_raw'] = a[i].reshape(-1) if a.ndim > 1 else a.reshape(-1)
        else:
            result['alpha_raw'] = np.zeros((100,), dtype=np.float32)

        if return_vertex_descriptor:
            verts_t = None
            for key in ("verts", "trans_verts"):
                if key in opdict:
                    verts_t = opdict[key]
                    break
            if verts_t is not None:
                verts_np = verts_t[i].detach().cpu().numpy().astype(np.float32)
                result['vertex_feat'] = compute_vertex_shape_descriptor(verts_np)
            else:
                result['vertex_feat'] = np.zeros((96,), dtype=np.float32)

        outputs.append(result)

    return outputs


def extract_coeffs_with_expression_batch(
    imgs_rgb_224: list,
    device: torch.device = torch.device("cpu"),
    model_tar: Optional[str] = None,
    coeff_clip: Optional[float] = 3.0,
    shape_scale: float = 1.0,
    expression_scale: float = 1.0,
) -> list:
    if not imgs_rgb_224:
        return []

    deca_model, _ = _get_deca(device=device, model_tar=model_tar)
    arr = []
    for img in imgs_rgb_224:
        if img.shape[:2] != (224, 224):
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        arr.append(img.astype(np.float32) / 255.0)
    ten = torch.from_numpy(np.stack(arr, axis=0)).permute(0, 3, 1, 2).to(device)

    with torch.no_grad():
        codedict = deca_model.encode(ten)

    outputs = []
    if isinstance(codedict, dict):
        alpha_all = codedict.get("shape", None)
        beta_all = codedict.get("exp", None)
        if alpha_all is None or beta_all is None:
            return [np.zeros((1,), dtype=np.float32) for _ in imgs_rgb_224]
        alpha_all = alpha_all.detach().cpu().numpy().astype(np.float32)
        beta_all = beta_all.detach().cpu().numpy().astype(np.float32)
        for i in range(alpha_all.shape[0]):
            alpha = alpha_all[i].reshape(-1)
            beta = beta_all[i].reshape(-1)
            if coeff_clip is not None and float(coeff_clip) > 0:
                alpha = np.clip(alpha, -float(coeff_clip), float(coeff_clip))
                beta = np.clip(beta, -float(coeff_clip), float(coeff_clip))
            alpha = alpha * float(shape_scale)
            beta = beta * float(expression_scale)
            coeff = np.concatenate([alpha, beta], axis=0).astype(np.float32)
            norm = float(np.linalg.norm(coeff))
            if norm > 1e-12:
                coeff = coeff / norm
            outputs.append(coeff)
    return outputs


def reconstruct_frontal_batch(
    imgs_rgb_224: list,
    device: torch.device = torch.device("cpu"),
    model_tar: Optional[str] = None,
    neutralize_exp: bool = False,
    neutralize_pose: bool = True,
) -> list:
    if not imgs_rgb_224:
        return []
    deca_model, _ = _get_deca(device=device, model_tar=model_tar)
    arr = []
    for img in imgs_rgb_224:
        if img.shape[:2] != (224, 224):
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        arr.append(img.astype(np.float32) / 255.0)
    ten = torch.from_numpy(np.stack(arr, axis=0)).permute(0, 3, 1, 2).to(device)
    with torch.no_grad():
        codedict = deca_model.encode(ten)
        codedict_render = _neutralize_codedict(
            codedict, neutralize_exp=neutralize_exp, neutralize_pose=neutralize_pose
        )
        decoded = deca_model.decode(codedict_render)
    if isinstance(decoded, tuple):
        opdict = decoded[0] if len(decoded) > 0 else {}
        visdict = decoded[1] if len(decoded) > 1 else {}
    elif isinstance(decoded, dict):
        opdict = decoded
        visdict = {}
    else:
        opdict = {}
        visdict = {}
    outputs = []
    for i, img in enumerate(imgs_rgb_224):
        outputs.append(_extract_frontal_bgr_at(opdict=opdict, visdict=visdict, img_rgb=img, idx=i))
    return outputs


def reconstruct_canonical_face(
    img_path: ImageInput,
    out_mesh_path: Optional[str] = None,
    out_frontal_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
    input_is_rgb: bool = False,
    return_frontal: bool = False,
    coeff_clip: Optional[float] = 3.0,
    shape_scale: float = 1.0,
    expression_scale: float = 1.0,
    model_tar: Optional[str] = None,
    neutralize_exp: bool = False,
    neutralize_pose: bool = True,
) -> Dict[str, np.ndarray]:
    img_rgb = _load_rgb_image(img_path, input_is_rgb=input_is_rgb)
    img_rgb_224 = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

    deca_model, _ = _get_deca(device=device, model_tar=model_tar)
    inp = _prepare_tensor_224(img_rgb, device=device)

    with torch.no_grad():
        codedict = deca_model.encode(inp)
        codedict_render = _neutralize_codedict(
            codedict, neutralize_exp=neutralize_exp, neutralize_pose=neutralize_pose
        )
        decoded = deca_model.decode(codedict_render)

    if isinstance(decoded, tuple):
        opdict = decoded[0] if len(decoded) > 0 else {}
        visdict = decoded[1] if len(decoded) > 1 else {}
    elif isinstance(decoded, dict):
        opdict = decoded
        visdict = {}
    else:
        opdict = {}
        visdict = {}

    verts_t = None
    for key in ("verts", "trans_verts"):
        if key in opdict:
            verts_t = opdict[key]
            break
    if verts_t is None:
        raise RuntimeError("Backend DECA-based khong tra ve vertices.")

    verts = verts_t[0].detach().cpu().numpy().astype(np.float32)
    center = verts.mean(axis=0, keepdims=True)
    verts = verts - center
    scale = float(np.max(np.linalg.norm(verts, axis=1)))
    if scale > 1e-8:
        verts = verts / scale * 100.0

    faces = _extract_faces(deca_model)
    vertex_colors = _sample_vertex_colors(verts, img_rgb_224)

    alpha = np.asarray([], dtype=np.float32)
    beta = np.asarray([], dtype=np.float32)
    detail = np.asarray([], dtype=np.float32)

    if isinstance(codedict, dict):
        if "shape" in codedict:
            alpha = codedict["shape"][0].detach().cpu().numpy().astype(np.float32).reshape(-1)
        if "exp" in codedict:
            beta = codedict["exp"][0].detach().cpu().numpy().astype(np.float32).reshape(-1)
        if "detail" in codedict:
            detail = codedict["detail"][0].detach().cpu().numpy().astype(np.float32).reshape(-1)

    if coeff_clip is not None and float(coeff_clip) > 0:
        alpha = np.clip(alpha, -float(coeff_clip), float(coeff_clip))
        beta = np.clip(beta, -float(coeff_clip), float(coeff_clip))
    alpha = alpha * float(shape_scale)
    beta = beta * float(expression_scale)

    frontal_bgr = _extract_frontal_bgr(opdict=opdict, visdict=visdict, img_rgb=img_rgb)
    gray = cv2.cvtColor(frontal_bgr, cv2.COLOR_BGR2GRAY)
    frontal_valid_ratio = float(np.mean(gray > 8))
    frontal_sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())

    if out_mesh_path is not None:
        os.makedirs(os.path.dirname(out_mesh_path) or ".", exist_ok=True)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh.visual.vertex_colors = vertex_colors
        mesh.export(out_mesh_path)

    if out_frontal_path is not None:
        os.makedirs(os.path.dirname(out_frontal_path) or ".", exist_ok=True)
        cv2.imwrite(out_frontal_path, frontal_bgr)

    params = alpha.astype(np.float32) if alpha.size > 0 else np.zeros((100,), dtype=np.float32)

    identity_feat = _extract_shape_feature(
        codedict, coeff_clip=coeff_clip, shape_scale=shape_scale, use_detail=True
    )

    pose = np.zeros((3, 4), dtype=np.float32)

    output: Dict[str, np.ndarray] = {
        "vertices": verts.astype(np.float32),
        "faces": faces.astype(np.int64),
        "params": params,            # shape-only (FIXED)
        "pose": pose,
        "alpha": alpha.astype(np.float32),   # shape codes
        "beta": beta.astype(np.float32),     # expression codes (lưu để debug, KHÔNG dùng cho identity)
        "detail": detail.astype(np.float32), # detail codes (NEW)
        "identity_feat": identity_feat,      # ready-to-use identity vector (NEW)
        "vertex_colors": vertex_colors.astype(np.uint8),
        "frontal_valid_ratio": np.asarray(frontal_valid_ratio, dtype=np.float32),
        "frontal_sharpness": np.asarray(frontal_sharpness, dtype=np.float32),
    }
    if return_frontal or out_frontal_path is not None:
        output["frontal_bgr"] = frontal_bgr
    return output
