import argparse
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .preprocessing import process_face_retina
from .reconstruct import reconstruct_image
from .recognition import ensure_arcface_session, arcface_embed, cosine_similarity


@dataclass
class PipelineOutputs:
    aligned_rgb_224: np.ndarray
    aligned_bgr_112: np.ndarray
    frontal_bgr: Optional[np.ndarray]
    embedding: np.ndarray
    embedding_2d: np.ndarray
    embedding_3d: np.ndarray
    out_dir: str


def run_pipeline(
    image_path: str,
    out_dir: str,
    device: str = 'cuda',
    allow_fallback: bool = True,
    coeff_clip: float = 3.0,
    shape_scale: float = 1.0,
    expression_scale: float = 1.0,
    model_tar: Optional[str] = None,
    mode: str = 'fused',
    fused_weight_2d: float = 0.8,
    align_size_3d: int = 224,
    align_template_scale_3d: float = 1.12,
    align_border_3d: str = 'reflect',
) -> PipelineOutputs:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f'Image not found: {image_path}')

    os.makedirs(out_dir, exist_ok=True)

    aligned_rgb_224 = process_face_retina(
        image_path,
        output_size=align_size_3d,
        allow_fallback=allow_fallback,
        template_scale=align_template_scale_3d,
        border_mode=align_border_3d,
    )

    aligned_rgb_112 = cv2.resize(aligned_rgb_224, (112, 112), interpolation=cv2.INTER_LINEAR)
    aligned_bgr_112 = cv2.cvtColor(aligned_rgb_112, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(out_dir, 'aligned_112.png'), aligned_bgr_112)

    recon = reconstruct_image(
        aligned_rgb_224,
        out_dir=out_dir,
        device=device,
        model_tar=model_tar,
        coeff_clip=coeff_clip,
        shape_scale=shape_scale,
        expression_scale=expression_scale,
    )

    frontal_bgr = recon.get('frontal_bgr') if isinstance(recon, dict) else None
    if frontal_bgr is None:
        frontal_bgr = cv2.resize(aligned_bgr_112, (224, 224), interpolation=cv2.INTER_LINEAR)

    sess = ensure_arcface_session()
    emb_2d = arcface_embed(sess, aligned_bgr_112)
    emb_3d = arcface_embed(sess, frontal_bgr)

    mode = (mode or 'fused').lower().strip()
    if mode not in ('2d_only', '3d_only', 'fused'):
        raise ValueError(f"Invalid mode: {mode}. Use 2d_only, 3d_only, fused.")

    if mode == '2d_only':
        embedding = emb_2d
    elif mode == '3d_only':
        embedding = emb_3d
    else:
        w2d = float(fused_weight_2d)
        w2d = max(0.0, min(1.0, w2d))
        w3d = 1.0 - w2d
        embedding = emb_2d * w2d + emb_3d * w3d
        embedding = embedding / (np.linalg.norm(embedding) + 1e-12)

    np.save(os.path.join(out_dir, 'embedding.npy'), embedding.astype(np.float32))

    return PipelineOutputs(
        aligned_rgb_224=aligned_rgb_224,
        aligned_bgr_112=aligned_bgr_112,
        frontal_bgr=frontal_bgr,
        embedding=embedding,
        embedding_2d=emb_2d,
        embedding_3d=emb_3d,
        out_dir=out_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='NCKH pipeline (DECA-based): preprocess -> reconstruct -> canonical -> ArcFace')
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--out-dir', default='outputs', help='Output directory')
    parser.add_argument('--compare', default=None, help='Optional second image to compare')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--no-fallback', action='store_true', help='Disable center-crop fallback when no face detected')
    parser.add_argument('--coeff-clip', type=float, default=3.0, help='Clip coeff for canonical reconstruction')
    parser.add_argument('--shape-scale', type=float, default=1.0, help='Scale shape coeff')
    parser.add_argument('--expression-scale', type=float, default=1.0, help='Scale expression coeff')
    parser.add_argument('--mode', default='fused', choices=['2d_only', '3d_only', 'fused'], help='Embedding mode')
    parser.add_argument('--fused-weight-2d', type=float, default=0.8, help='Weight for 2D in fused mode')
    parser.add_argument('--align-size-3d', type=int, default=224, help='Alignment size for DECA-based reconstruction')
    parser.add_argument('--align-template-3d', type=float, default=1.12, help='Template scale for DECA-based alignment')
    parser.add_argument('--align-border-3d', default='reflect', help='Border mode for DECA-based alignment')
    parser.add_argument('--model-tar', default=None, help='Path to DECA-based checkpoint (.tar)')
    args = parser.parse_args()

    model_tar = (
        args.model_tar
        or os.environ.get('FACE3D_MODEL_TAR', '').strip()
        or os.environ.get('DECA_MODEL_TAR', '').strip()
        or None
    )
    if model_tar:
        print(f"[INFO] FACE3D_MODEL_TAR={model_tar}")
        print(f"[INFO] checkpoint exists: {os.path.isfile(model_tar)}")

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    result = run_pipeline(
        args.image,
        out_dir,
        device=args.device,
        allow_fallback=not args.no_fallback,
        coeff_clip=args.coeff_clip,
        shape_scale=args.shape_scale,
        expression_scale=args.expression_scale,
        model_tar=model_tar,
        mode=args.mode,
        fused_weight_2d=args.fused_weight_2d,
        align_size_3d=args.align_size_3d,
        align_template_scale_3d=args.align_template_3d,
        align_border_3d=args.align_border_3d,
    )

    print('Aligned:', os.path.join(out_dir, 'aligned_112.png'))
    print('Frontal:', os.path.join(out_dir, 'canonical_frontal.png'))
    print('Mesh:', os.path.join(out_dir, 'canonical.obj'))
    print('Embedding:', os.path.join(out_dir, 'embedding.npy'))

    if args.compare:
        out_dir_cmp = os.path.join(out_dir, 'compare')
        result_cmp = run_pipeline(
            args.compare,
            out_dir_cmp,
            device=args.device,
            allow_fallback=not args.no_fallback,
            coeff_clip=args.coeff_clip,
            shape_scale=args.shape_scale,
            expression_scale=args.expression_scale,
            model_tar=model_tar,
            mode=args.mode,
            fused_weight_2d=args.fused_weight_2d,
            align_size_3d=args.align_size_3d,
            align_template_scale_3d=args.align_template_3d,
            align_border_3d=args.align_border_3d,
        )
        sim = cosine_similarity(result.embedding, result_cmp.embedding)
        sim2 = cosine_similarity(result.embedding_2d, result_cmp.embedding_2d)
        sim3 = cosine_similarity(result.embedding_3d, result_cmp.embedding_3d)
        print(f'Cosine similarity: {sim:.4f}')
        print(f'Cosine similarity (2D): {sim2:.4f}')
        print(f'Cosine similarity (3D): {sim3:.4f}')


if __name__ == '__main__':
    main()
