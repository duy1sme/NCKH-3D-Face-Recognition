import argparse
import os
import sys
from pathlib import Path


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from core.pipeline import run_pipeline
from core.recognition import cosine_similarity


def _resolve_path(p: str, project_root: Path) -> str:
    cand = Path(p)
    if cand.is_absolute():
        return str(cand)
    if cand.exists():
        return str(cand.resolve())
    return str((project_root / cand).resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full pipeline for a pair of images and compute cosine similarity."
    )
    parser.add_argument("--image-a", required=True, help="First image path")
    parser.add_argument("--image-b", required=True, help="Second image path")
    parser.add_argument(
        "--out-dir",
        default=os.path.join("outputs", "demo_pair"),
        help="Output folder",
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--model-tar", default=None, help="Path to team DECA-based checkpoint (.tar)")
    parser.add_argument(
        "--mode",
        default="fused",
        choices=["2d_only", "3d_only", "fused"],
        help="Embedding mode",
    )
    parser.add_argument(
        "--fused-weight-2d",
        type=float,
        default=0.8,
        help="2D weight in fused mode",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable center-crop fallback when detector misses",
    )
    parser.add_argument("--align-size-3d", type=int, default=224, help="Alignment size for 3D reconstruction")
    parser.add_argument("--align-template-3d", type=float, default=1.12, help="Template scale for 3D alignment")
    parser.add_argument("--align-border-3d", default="reflect", help="Border mode for 3D alignment")
    args = parser.parse_args()
    project_root = Path(ROOT_DIR)

    image_a = _resolve_path(args.image_a, project_root)
    image_b = _resolve_path(args.image_b, project_root)
    out_dir = _resolve_path(args.out_dir, project_root)

    model_tar = (
        args.model_tar
        or os.environ.get("FACE3D_MODEL_TAR", "").strip()
        or os.environ.get("DECA_MODEL_TAR", "").strip()
        or None
    )
    if model_tar:
        model_tar = _resolve_path(model_tar, project_root)

    out_a = os.path.join(out_dir, "A")
    out_b = os.path.join(out_dir, "B")

    res_a = run_pipeline(
        image_path=image_a,
        out_dir=out_a,
        device=args.device,
        allow_fallback=not args.no_fallback,
        model_tar=model_tar,
        mode=args.mode,
        fused_weight_2d=args.fused_weight_2d,
        align_size_3d=args.align_size_3d,
        align_template_scale_3d=args.align_template_3d,
        align_border_3d=args.align_border_3d,
    )
    res_b = run_pipeline(
        image_path=image_b,
        out_dir=out_b,
        device=args.device,
        allow_fallback=not args.no_fallback,
        model_tar=model_tar,
        mode=args.mode,
        fused_weight_2d=args.fused_weight_2d,
        align_size_3d=args.align_size_3d,
        align_template_scale_3d=args.align_template_3d,
        align_border_3d=args.align_border_3d,
    )

    sim = cosine_similarity(res_a.embedding, res_b.embedding)
    sim2d = cosine_similarity(res_a.embedding_2d, res_b.embedding_2d)
    sim3d = cosine_similarity(res_a.embedding_3d, res_b.embedding_3d)

    print("Done.")
    print(f"Cosine ({args.mode}): {sim:.4f}")
    print(f"Cosine (2D only): {sim2d:.4f}")
    print(f"Cosine (3D only): {sim3d:.4f}")
    print("Outputs:", os.path.abspath(out_dir))


if __name__ == "__main__":
    main()
