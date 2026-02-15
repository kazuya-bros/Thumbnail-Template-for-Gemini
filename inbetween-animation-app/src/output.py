"""
Output module for exporting animation frames.

Supports: PNG sequence, GIF, APNG, MP4.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional


def save_frames_as_png(
    frames: list[np.ndarray],
    output_dir: str,
    prefix: str = "frame",
) -> list[str]:
    """
    Save frames as a numbered PNG sequence.

    Args:
        frames: List of BGR images
        output_dir: Directory to save frames
        prefix: Filename prefix

    Returns:
        List of saved file paths
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    paths = []
    digits = len(str(len(frames)))
    for i, frame in enumerate(frames):
        filename = f"{prefix}_{str(i).zfill(digits)}.png"
        filepath = out_path / filename
        cv2.imwrite(str(filepath), frame)
        paths.append(str(filepath))

    print(f"[Output] Saved {len(frames)} PNG frames to {output_dir}")
    return paths


def save_frames_as_gif(
    frames: list[np.ndarray],
    output_path: str,
    fps: int = 24,
    loop: int = 0,
) -> str:
    """
    Save frames as an animated GIF.

    Args:
        frames: List of BGR images
        output_path: Output GIF file path
        fps: Frames per second
        loop: Number of loops (0 = infinite)

    Returns:
        Output file path
    """
    import imageio.v3 as iio

    # Convert BGR to RGB for imageio
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]

    duration_ms = 1000.0 / fps

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    iio.imwrite(
        output_path,
        rgb_frames,
        extension=".gif",
        duration=duration_ms,
        loop=loop,
    )

    print(f"[Output] Saved GIF: {output_path} ({len(frames)} frames, {fps} fps)")
    return output_path


def save_frames_as_apng(
    frames: list[np.ndarray],
    output_path: str,
    fps: int = 24,
    loop: int = 0,
) -> str:
    """
    Save frames as an animated PNG (APNG).
    Better quality than GIF (supports full color and alpha).

    Args:
        frames: List of BGR images
        output_path: Output APNG file path
        fps: Frames per second
        loop: Number of loops (0 = infinite)

    Returns:
        Output file path
    """
    from PIL import Image

    pil_frames = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        pil_frames.append(Image.fromarray(rgb))

    duration_ms = int(1000 / fps)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=loop,
    )

    print(f"[Output] Saved APNG: {output_path} ({len(frames)} frames, {fps} fps)")
    return output_path


def save_frames_as_mp4(
    frames: list[np.ndarray],
    output_path: str,
    fps: int = 24,
) -> str:
    """
    Save frames as an MP4 video.

    Args:
        frames: List of BGR images
        output_path: Output MP4 file path
        fps: Frames per second

    Returns:
        Output file path
    """
    if not frames:
        raise ValueError("No frames to save")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in frames:
        writer.write(frame)

    writer.release()

    print(f"[Output] Saved MP4: {output_path} ({len(frames)} frames, {fps} fps)")
    return output_path
