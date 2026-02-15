"""
RIFE-based inbetween generation.

Uses the Practical-RIFE model for high-quality neural frame interpolation.
Requires GPU for reasonable performance.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional


class RIFEInterpolator:
    """Frame interpolation using RIFE (Real-Time Intermediate Flow Estimation)."""

    def __init__(self, model_dir: Optional[str] = None, device: str = "cuda"):
        self.device = device
        self.model_dir = model_dir or str(Path(__file__).parent.parent / "models" / "rife")
        self.model = None
        self._loaded = False

    def load_model(self):
        """Load RIFE model. Lazy-loaded on first use."""
        if self._loaded:
            return

        try:
            import torch
            import sys

            rife_path = Path(self.model_dir)
            if not rife_path.exists():
                print(f"[RIFE] Model directory not found: {rife_path}")
                print("[RIFE] Please download Practical-RIFE and place it in the models/rife directory.")
                print("[RIFE]   git clone https://github.com/hzwer/Practical-RIFE models/rife")
                self._loaded = True
                return

            # Add RIFE to path for importing
            sys.path.insert(0, str(rife_path))

            from model.RIFE import Model

            self.model = Model()
            self.model.load_model(str(rife_path / "train_log"), -1)
            self.model.eval()
            self._loaded = True
            print("[RIFE] Model loaded successfully")

        except ImportError as e:
            print(f"[RIFE] Failed to import RIFE model: {e}")
            print("[RIFE] Falling back to OpenCV optical flow interpolation.")
            self.model = None
            self._loaded = True
        except Exception as e:
            print(f"[RIFE] Failed to load model: {e}")
            self.model = None
            self._loaded = True

    def interpolate(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        timestep: float = 0.5,
    ) -> np.ndarray:
        """
        Interpolate between two images at the given timestep.

        Args:
            image_a: First image (BGR, uint8)
            image_b: Second image (BGR, uint8)
            timestep: Interpolation position. 0.0 = image_a, 1.0 = image_b

        Returns:
            Interpolated image (BGR, uint8)
        """
        self.load_model()

        if self.model is not None:
            return self._interpolate_rife(image_a, image_b, timestep)
        else:
            return self._interpolate_optical_flow(image_a, image_b, timestep)

    def _interpolate_rife(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        timestep: float,
    ) -> np.ndarray:
        """Interpolate using RIFE model."""
        import torch

        h, w = image_a.shape[:2]

        # RIFE expects RGB float tensors
        img0 = cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img1 = cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Convert to tensors (B, C, H, W)
        img0_t = torch.from_numpy(img0.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        img1_t = torch.from_numpy(img1.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        # Pad to multiple of 32 (RIFE requirement)
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            img0_t = torch.nn.functional.pad(img0_t, (0, pad_w, 0, pad_h), mode="replicate")
            img1_t = torch.nn.functional.pad(img1_t, (0, pad_w, 0, pad_h), mode="replicate")

        with torch.no_grad():
            result_t = self.model.inference(img0_t, img1_t, timestep=timestep)

        # Remove padding and convert back
        result = result_t.squeeze().cpu().numpy().transpose(1, 2, 0)[:h, :w]
        result = (result * 255).clip(0, 255).astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        return result

    def _interpolate_optical_flow(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        timestep: float,
    ) -> np.ndarray:
        """
        Fallback interpolation using OpenCV optical flow.
        Not as good as RIFE but works without the model.
        """
        gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow (Farneback)
        flow_ab = cv2.calcOpticalFlowFarneback(
            gray_a, gray_b, None,
            pyr_scale=0.5, levels=5, winsize=15,
            iterations=3, poly_n=7, poly_sigma=1.5, flags=0,
        )

        h, w = image_a.shape[:2]

        # Create the warped intermediate frame
        # Forward warp from A towards B
        flow_t = flow_ab * timestep
        map_x = np.float32(np.tile(np.arange(w), (h, 1))) + flow_t[:, :, 0]
        map_y = np.float32(np.tile(np.arange(h).reshape(-1, 1), (1, w))) + flow_t[:, :, 1]

        warped_a = cv2.remap(image_a, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # Backward warp from B towards A
        flow_t_inv = flow_ab * (timestep - 1.0)
        map_x_inv = np.float32(np.tile(np.arange(w), (h, 1))) + flow_t_inv[:, :, 0]
        map_y_inv = np.float32(np.tile(np.arange(h).reshape(-1, 1), (1, w))) + flow_t_inv[:, :, 1]

        warped_b = cv2.remap(image_b, map_x_inv, map_y_inv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # Blend the two warped images
        result = cv2.addWeighted(warped_a, 1.0 - timestep, warped_b, timestep, 0)

        return result


def generate_inbetween_rife(
    image_a: np.ndarray,
    image_b: np.ndarray,
    mask: np.ndarray,
    ratio: float,
    interpolator: Optional[RIFEInterpolator] = None,
) -> np.ndarray:
    """
    Generate an inbetween frame using RIFE within a masked region.

    Args:
        image_a: Source image (BGR, uint8)
        image_b: Target image (BGR, uint8)
        mask: Binary mask for the region to interpolate (H, W), uint8
        ratio: Interpolation ratio. 0.0 = image_a, 1.0 = image_b
        interpolator: RIFEInterpolator instance (will create one if None)

    Returns:
        Composited image with interpolated region (BGR, uint8)
    """
    if interpolator is None:
        interpolator = RIFEInterpolator()

    ratio = np.clip(ratio, 0.0, 1.0)

    if ratio < 0.01:
        return image_a.copy()
    if ratio > 0.99:
        return image_b.copy()

    # Get bounding box of the mask
    coords = cv2.findNonZero(mask)
    if coords is None:
        return image_a.copy()

    x, y, bw, bh = cv2.boundingRect(coords)

    # Add padding around the crop
    pad = 32
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(image_a.shape[1], x + bw + pad)
    y2 = min(image_a.shape[0], y + bh + pad)

    # Crop the region from both images
    crop_a = image_a[y1:y2, x1:x2].copy()
    crop_b = image_b[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]

    # Interpolate the cropped region
    interpolated_crop = interpolator.interpolate(crop_a, crop_b, ratio)

    # Create soft mask for blending
    soft_mask = crop_mask.astype(np.float32) / 255.0
    ksize = 15
    soft_mask = cv2.GaussianBlur(soft_mask, (ksize, ksize), 0)
    soft_mask_3ch = np.stack([soft_mask] * 3, axis=-1)

    # Composite back
    result = image_a.copy()
    region = result[y1:y2, x1:x2].astype(np.float32)
    interpolated = interpolated_crop.astype(np.float32)
    composited = region * (1.0 - soft_mask_3ch) + interpolated * soft_mask_3ch
    result[y1:y2, x1:x2] = np.clip(composited, 0, 255).astype(np.uint8)

    return result
