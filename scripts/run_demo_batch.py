import argparse
import csv
import os
import sys
from pathlib import Path


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from core.pipeline import run_pipeline


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _resolve_path(p: str, project_root: Path) -> Path:
    cand = Path(p)
    if cand.is_absolute():
        return cand
    if cand.exists():
        return cand.resolve()
    return (project_root / cand).resolve()


def collect_images(root: Path) -> list[Path]:
    items = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            items.append(p)
    return sorted(items)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full pipeline for all images in a folder."
    )
    parser.add_argument("--image-dir", required=True, help="Input image directory")
    parser.add_argument(
        "--out-dir",
        default=os.path.join("outputs", "demo_batch"),
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
        "--max-images",
        type=int,
        default=0,
        help="Limit number of images (0 means all)",
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

    image_dir = _resolve_path(args.image_dir, project_root)
    out_dir = _resolve_path(args.out_dir, project_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tar = (
        args.model_tar
        or os.environ.get("FACE3D_MODEL_TAR", "").strip()
        or os.environ.get("DECA_MODEL_TAR", "").strip()
        or None
    )
    if model_tar:
        model_tar = str(_resolve_path(model_tar, project_root))

    if not image_dir.exists():
        raise FileNotFoundError(str(image_dir))

    images = collect_images(image_dir)
    if args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        raise RuntimeError("No image found in input directory.")

    rows = []
    total = len(images)
    for idx, p in enumerate(images, start=1):
        rel = p.relative_to(image_dir)
        sample_out = out_dir / rel.with_suffix("")
        sample_out.mkdir(parents=True, exist_ok=True)
        try:
            run_pipeline(
                image_path=str(p),
                out_dir=str(sample_out),
                device=args.device,
                allow_fallback=not args.no_fallback,
                model_tar=model_tar,
                mode=args.mode,
                fused_weight_2d=args.fused_weight_2d,
                align_size_3d=args.align_size_3d,
                align_template_scale_3d=args.align_template_3d,
                align_border_3d=args.align_border_3d,
            )
            status = "ok"
            err = ""
        except Exception as ex:
            status = "fail"
            err = str(ex).replace("\n", " ")[:500]

        rows.append([str(rel).replace("\\", "/"), status, err])
        if idx % 20 == 0 or idx == total:
            print(f"[{idx}/{total}] done")

    report = out_dir / "batch_report.csv"
    with report.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "status", "error"])
        wr.writerows(rows)

    ok = sum(1 for r in rows if r[1] == "ok")
    print("Done.")
    print(f"Success: {ok}/{len(rows)}")
    print("Report:", report)


if __name__ == "__main__":
    main()
