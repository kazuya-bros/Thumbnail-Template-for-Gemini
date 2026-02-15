"""
Anime face segmentation module.

Provides multiple backends for extracting eye/mouth masks from anime illustrations:
1. Anime-Face-Segmentation (MobileNetV2) - Primary
2. Simple color/edge-based fallback
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

# Segmentation class indices for Anime-Face-Segmentation model
SEG_CLASSES = {
    "background": 0,
    "hair": 1,
    "eye": 2,
    "mouth": 3,
    "face": 4,
    "skin": 5,
    "clothes": 6,
}


class AnimeSegmenter:
    """Anime face segmentation using MobileNetV2-based model."""

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        self.device = device
        self.model = None
        self.model_path = model_path
        self._loaded = False

    def load_model(self):
        """Load the segmentation model. Lazy-loaded on first use."""
        if self._loaded:
            return

        try:
            import torch
            import torchvision.models.segmentation as seg_models

            # DeepLabV3 with MobileNetV2 backbone
            self.model = seg_models.deeplabv3_mobilenet_v3_large(num_classes=7)

            if self.model_path and Path(self.model_path).exists():
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"[Segmentation] Loaded model from {self.model_path}")
            else:
                print("[Segmentation] No pretrained anime model found. Using fallback segmentation.")
                self.model = None
                self._loaded = True
                return

            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
        except Exception as e:
            print(f"[Segmentation] Failed to load model: {e}. Using fallback.")
            self.model = None
            self._loaded = True

    def segment(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """
        Segment an anime face image into parts.

        Args:
            image: BGR image (H, W, 3), uint8

        Returns:
            Dict of binary masks {class_name: mask} where mask is (H, W), uint8, 0 or 255
        """
        self.load_model()

        if self.model is not None:
            return self._segment_with_model(image)
        else:
            return self._segment_fallback(image)

    def _segment_with_model(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Segment using the neural network model."""
        import torch
        from torchvision import transforms

        h, w = image.shape[:2]

        # Preprocess: resize to 512x512, normalize
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = transform(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)["out"]
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # Resize prediction back to original size
        pred_resized = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        masks = {}
        for class_name, class_idx in SEG_CLASSES.items():
            mask = (pred_resized == class_idx).astype(np.uint8) * 255
            masks[class_name] = mask

        return masks

    def _segment_fallback(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """
        Fallback segmentation using traditional CV techniques.
        Detects face region, then estimates eye/mouth positions heuristically.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use face center assumption: face is roughly centered
        # Eye region: upper 30-50% of image height, central 70% width
        # Mouth region: 55-75% of image height, central 40% width
        eye_mask = np.zeros((h, w), dtype=np.uint8)
        mouth_mask = np.zeros((h, w), dtype=np.uint8)
        face_mask = np.zeros((h, w), dtype=np.uint8)

        # Heuristic face region (center 60%)
        face_y1, face_y2 = int(h * 0.15), int(h * 0.85)
        face_x1, face_x2 = int(w * 0.15), int(w * 0.85)
        face_mask[face_y1:face_y2, face_x1:face_x2] = 255

        # Eye region heuristic
        eye_y1, eye_y2 = int(h * 0.28), int(h * 0.50)
        eye_x1, eye_x2 = int(w * 0.12), int(w * 0.88)

        # Detect high-contrast regions within eye area (anime eyes have strong features)
        eye_roi = gray[eye_y1:eye_y2, eye_x1:eye_x2]
        # Use edge detection to find eye boundaries
        edges = cv2.Canny(eye_roi, 50, 150)
        # Dilate to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        # Find contours and create mask from large contours (likely eyes)
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        eye_roi_mask = np.zeros_like(eye_roi)
        min_area = (eye_roi.shape[0] * eye_roi.shape[1]) * 0.03  # At least 3% of ROI
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                cv2.drawContours(eye_roi_mask, [cnt], -1, 255, -1)

        if np.sum(eye_roi_mask) > 0:
            eye_mask[eye_y1:eye_y2, eye_x1:eye_x2] = eye_roi_mask
        else:
            # Fallback: simple rectangle regions for left and right eyes
            left_eye_x1, left_eye_x2 = int(w * 0.15), int(w * 0.42)
            right_eye_x1, right_eye_x2 = int(w * 0.58), int(w * 0.85)
            eye_mask[eye_y1:eye_y2, left_eye_x1:left_eye_x2] = 255
            eye_mask[eye_y1:eye_y2, right_eye_x1:right_eye_x2] = 255

        # Mouth region heuristic
        mouth_y1, mouth_y2 = int(h * 0.58), int(h * 0.75)
        mouth_x1, mouth_x2 = int(w * 0.32), int(w * 0.68)

        mouth_roi = gray[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
        edges_mouth = cv2.Canny(mouth_roi, 30, 100)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        edges_mouth_dilated = cv2.dilate(edges_mouth, kernel_small, iterations=2)
        contours_mouth, _ = cv2.findContours(edges_mouth_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mouth_roi_mask = np.zeros_like(mouth_roi)
        min_mouth_area = (mouth_roi.shape[0] * mouth_roi.shape[1]) * 0.05
        for cnt in contours_mouth:
            if cv2.contourArea(cnt) > min_mouth_area:
                cv2.drawContours(mouth_roi_mask, [cnt], -1, 255, -1)

        if np.sum(mouth_roi_mask) > 0:
            mouth_mask[mouth_y1:mouth_y2, mouth_x1:mouth_x2] = mouth_roi_mask
        else:
            mouth_mask[mouth_y1:mouth_y2, mouth_x1:mouth_x2] = 255

        # Dilate masks slightly for safety margin
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        eye_mask = cv2.dilate(eye_mask, dilate_kernel, iterations=1)
        mouth_mask = cv2.dilate(mouth_mask, dilate_kernel, iterations=1)

        return {
            "eye": eye_mask,
            "mouth": mouth_mask,
            "face": face_mask,
            "background": cv2.bitwise_not(face_mask),
        }

    def get_combined_mask(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        part: str,
        dilate_px: int = 10,
    ) -> np.ndarray:
        """
        Get a combined mask from both images for a given part.
        Takes the union of masks from both images and dilates for safety.

        Args:
            image_a: First image (BGR)
            image_b: Second image (BGR)
            part: "eye" or "mouth"
            dilate_px: Dilation amount in pixels

        Returns:
            Combined binary mask (H, W), uint8, 0 or 255
        """
        masks_a = self.segment(image_a)
        masks_b = self.segment(image_b)

        mask_a = masks_a.get(part, np.zeros(image_a.shape[:2], dtype=np.uint8))
        mask_b = masks_b.get(part, np.zeros(image_b.shape[:2], dtype=np.uint8))

        # Union of both masks
        combined = cv2.bitwise_or(mask_a, mask_b)

        # Dilate for safety margin
        if dilate_px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
            combined = cv2.dilate(combined, kernel, iterations=1)

        return combined
