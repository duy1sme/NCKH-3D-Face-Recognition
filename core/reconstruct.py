import argparse
import os
from typing import Optional

import cv2
import numpy as np
import torch

from .canonical import reconstruct_canonical_face, reconstruct_frontal_batch


def reconstruct_image(
    img_rgb_224: np.ndarray,
    out_dir: str,
    device: str = 'cuda',
    model_tar: Optional[str] = None,
    coeff_clip: float = 3.0,
    shape_scale: float = 1.0,
    expression_scale: float = 1.0,
    save_outputs: bool = True,
) -> dict:
    if save_outputs:
        os.makedirs(out_dir, exist_ok=True)
        mesh_path = os.path.join(out_dir, 'canonical.obj')
        frontal_path = os.path.join(out_dir, 'canonical_frontal.png')
    else:
        mesh_path = None
        frontal_path = None

    device_obj = torch.device(device if torch.cuda.is_available() and device != 'cpu' else 'cpu')
    recon = reconstruct_canonical_face(
        img_rgb_224,
        out_mesh_path=mesh_path,
        out_frontal_path=frontal_path,
        device=device_obj,
        input_is_rgb=True,
        return_frontal=True,
        coeff_clip=coeff_clip,
        shape_scale=shape_scale,
        expression_scale=expression_scale,
        model_tar=model_tar,
    )
    return recon


def reconstruct_frontal_images(
    imgs_rgb_224: list[np.ndarray],
    device: str = 'cuda',
    model_tar: Optional[str] = None,
) -> list[np.ndarray]:
    device_obj = torch.device(device if torch.cuda.is_available() and device != 'cpu' else 'cpu')
    return reconstruct_frontal_batch(
        imgs_rgb_224,
        device=device_obj,
        model_tar=model_tar,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Reconstruct canonical face using a DECA-based backend')
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--out-dir', default='outputs/reconstruct', help='Output directory')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--model-tar', default=None, help='Path to DECA-based checkpoint (.tar)')
    args = parser.parse_args()

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(args.image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

    model_tar = (
        args.model_tar
        or os.environ.get('FACE3D_MODEL_TAR', '').strip()
        or os.environ.get('DECA_MODEL_TAR', '').strip()
        or None
    )

    recon = reconstruct_image(
        img_rgb,
        out_dir=args.out_dir,
        device=args.device,
        model_tar=model_tar,
    )

    if isinstance(recon, dict) and 'frontal_bgr' in recon:
        print('Frontal:', os.path.join(args.out_dir, 'canonical_frontal.png'))
    print('Mesh:', os.path.join(args.out_dir, 'canonical.obj'))


if __name__ == '__main__':
    main()
