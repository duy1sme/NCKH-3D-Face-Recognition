import argparse
import os
from pathlib import Path


def main() -> None:
    this_dir = Path(__file__).resolve().parent
    project_root = this_dir.parent
    default_model_rel = os.path.join("models", "team_face3d_model.tar")
    default_model_abs = str((project_root / "models" / "team_face3d_model.tar").resolve())

    parser = argparse.ArgumentParser(
        description="Guide script to prepare external DECA-based backbone/checkpoint for demo."
    )
    parser.add_argument(
        "--model-path",
        default=default_model_rel,
        help="Expected local team checkpoint path",
    )
    args = parser.parse_args()

    print("Model setup checklist")
    print("=====================")
    print(f"Project root: {project_root}")
    print("1) Team DECA-based checkpoint:")
    print(f"   - Relative path (repo): {args.model_path}")
    print(f"   - Absolute path sample : {default_model_abs}")
    print("   - Or set env: FACE3D_MODEL_TAR=<PROJECT_ROOT>/models/team_face3d_model.tar")
    print("   - Backward-compatible env: DECA_MODEL_TAR=<PROJECT_ROOT>/models/team_face3d_model.tar")
    print("")
    print("2) ArcFace (InsightFace buffalo_l):")
    print("   - First run any demo script once.")
    print("   - InsightFace will auto-download buffalo_l into ~/.insightface/models/")
    print("")
    print("3) DECA-based backbone repo path:")
    print("   - Set env: FACE3D_BACKBONE_PATH=<PATH_TO_DECA_MASTER>")
    print("   - Backward-compatible env: DECA_REPO_PATH=<PATH_TO_DECA_MASTER>")
    print("")
    print("Done.")


if __name__ == "__main__":
    main()
