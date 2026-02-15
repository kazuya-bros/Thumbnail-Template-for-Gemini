"""
CLI interface for the inbetween animation tool.

Usage:
    # Generate a single inbetween frame at 50% interpolation
    python -m src.cli single --image-a samples/input/eyes_open.png \
                             --image-b samples/input/eyes_closed.png \
                             --eye-ratio 0.5 --mouth-ratio 0.5 \
                             --output samples/output/inbetween_50.png

    # Generate multiple inbetween frames
    python -m src.cli batch --image-a samples/input/eyes_open.png \
                            --image-b samples/input/eyes_closed.png \
                            --steps 4 \
                            --output-dir samples/output/batch

    # Generate a blink+talk animation
    python -m src.cli animate --image-a samples/input/eyes_open.png \
                              --image-b samples/input/eyes_closed.png \
                              --fps 24 --format gif \
                              --output samples/output/animation.gif

    # Visualize segmentation masks
    python -m src.cli masks --image samples/input/eyes_open.png \
                            --output samples/output/masks.png
"""

import sys
import click
import cv2
import numpy as np
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import (
    InbetweenPipeline,
    InbetweenMode,
    InbetweenParams,
    PipelineConfig,
    AnimationKeyframe,
    create_blink_talk_animation,
)
from src.output import (
    save_frames_as_png,
    save_frames_as_gif,
    save_frames_as_apng,
    save_frames_as_mp4,
)


@click.group()
@click.option("--mode", type=click.Choice(["blend", "poisson", "rife", "gemini", "hybrid"]),
              default="blend", help="Inbetween generation mode")
@click.option("--device", default="cuda", help="PyTorch device (cuda/cpu)")
@click.option("--gemini-key", envvar="GEMINI_API_KEY", default=None, help="Gemini API key")
@click.pass_context
def cli(ctx, mode, device, gemini_key):
    """Inbetween Animation Generator - Generate smooth animation frames from two keyframes."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = PipelineConfig(
        mode=InbetweenMode(mode),
        device=device,
        gemini_api_key=gemini_key,
    )


@cli.command()
@click.option("--image-a", required=True, type=click.Path(exists=True), help="Image A (eyes open, mouth closed)")
@click.option("--image-b", required=True, type=click.Path(exists=True), help="Image B (eyes closed, mouth open)")
@click.option("--eye-ratio", default=0.5, type=float, help="Eye closure ratio (0=open, 1=closed)")
@click.option("--mouth-ratio", default=0.5, type=float, help="Mouth opening ratio (0=closed, 1=open)")
@click.option("--output", "-o", required=True, help="Output image path")
@click.pass_context
def single(ctx, image_a, image_b, eye_ratio, mouth_ratio, output):
    """Generate a single inbetween frame."""
    config = ctx.obj["config"]
    pipeline = InbetweenPipeline(config)

    img_a = pipeline.load_image(image_a)
    img_b = pipeline.load_image(image_b)

    click.echo(f"Generating inbetween frame (eye={eye_ratio}, mouth={mouth_ratio})...")

    params = InbetweenParams(eye_ratio=eye_ratio, mouth_ratio=mouth_ratio)
    result = pipeline.generate_frame(img_a, img_b, params)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output, result)
    click.echo(f"Saved: {output}")


@cli.command()
@click.option("--image-a", required=True, type=click.Path(exists=True), help="Image A (eyes open, mouth closed)")
@click.option("--image-b", required=True, type=click.Path(exists=True), help="Image B (eyes closed, mouth open)")
@click.option("--steps", default=4, type=int, help="Number of intermediate steps (e.g., 4 = 0%, 25%, 50%, 75%, 100%)")
@click.option("--eye-only", is_flag=True, help="Only vary eye ratio (mouth stays at 0)")
@click.option("--mouth-only", is_flag=True, help="Only vary mouth ratio (eye stays at 0)")
@click.option("--output-dir", "-o", required=True, help="Output directory for frames")
@click.pass_context
def batch(ctx, image_a, image_b, steps, eye_only, mouth_only, output_dir):
    """Generate multiple inbetween frames at evenly spaced ratios."""
    config = ctx.obj["config"]
    pipeline = InbetweenPipeline(config)

    img_a = pipeline.load_image(image_a)
    img_b = pipeline.load_image(image_b)

    # Pre-compute masks
    eye_mask, mouth_mask = pipeline.get_masks(img_a, img_b)

    ratios = [i / steps for i in range(steps + 1)]
    click.echo(f"Generating {len(ratios)} frames at ratios: {ratios}")

    frames = []
    for i, ratio in enumerate(ratios):
        eye_r = ratio if not mouth_only else 0.0
        mouth_r = ratio if not eye_only else 0.0

        params = InbetweenParams(eye_ratio=eye_r, mouth_ratio=mouth_r)
        frame = pipeline.generate_frame(img_a, img_b, params, eye_mask, mouth_mask)
        frames.append(frame)
        click.echo(f"  [{i + 1}/{len(ratios)}] eye={eye_r:.2f}, mouth={mouth_r:.2f}")

    save_frames_as_png(frames, output_dir)
    click.echo(f"Done! {len(frames)} frames saved to {output_dir}")


@cli.command()
@click.option("--image-a", required=True, type=click.Path(exists=True), help="Image A (eyes open, mouth closed)")
@click.option("--image-b", required=True, type=click.Path(exists=True), help="Image B (eyes closed, mouth open)")
@click.option("--fps", default=24, type=int, help="Frames per second")
@click.option("--format", "fmt", type=click.Choice(["gif", "apng", "mp4", "png"]),
              default="gif", help="Output format")
@click.option("--output", "-o", required=True, help="Output file/directory path")
@click.option("--preset", type=click.Choice(["blink_talk", "blink", "talk"]),
              default="blink_talk", help="Animation preset")
@click.pass_context
def animate(ctx, image_a, image_b, fps, fmt, output, preset):
    """Generate a full animation sequence."""
    config = ctx.obj["config"]
    pipeline = InbetweenPipeline(config)

    img_a = pipeline.load_image(image_a)
    img_b = pipeline.load_image(image_b)

    # Select animation preset
    if preset == "blink_talk":
        keyframes = create_blink_talk_animation()
    elif preset == "blink":
        keyframes = [
            AnimationKeyframe(time=0.0, eye_ratio=0.0, mouth_ratio=0.0),
            AnimationKeyframe(time=0.1, eye_ratio=0.5, mouth_ratio=0.0),
            AnimationKeyframe(time=0.15, eye_ratio=1.0, mouth_ratio=0.0),
            AnimationKeyframe(time=0.2, eye_ratio=0.5, mouth_ratio=0.0),
            AnimationKeyframe(time=0.3, eye_ratio=0.0, mouth_ratio=0.0),
            AnimationKeyframe(time=0.8, eye_ratio=0.0, mouth_ratio=0.0),
        ]
    elif preset == "talk":
        keyframes = [
            AnimationKeyframe(time=0.0, eye_ratio=0.0, mouth_ratio=0.0),
            AnimationKeyframe(time=0.1, eye_ratio=0.0, mouth_ratio=0.5),
            AnimationKeyframe(time=0.2, eye_ratio=0.0, mouth_ratio=1.0),
            AnimationKeyframe(time=0.35, eye_ratio=0.0, mouth_ratio=0.5),
            AnimationKeyframe(time=0.45, eye_ratio=0.0, mouth_ratio=1.0),
            AnimationKeyframe(time=0.6, eye_ratio=0.0, mouth_ratio=0.3),
            AnimationKeyframe(time=0.75, eye_ratio=0.0, mouth_ratio=0.0),
            AnimationKeyframe(time=1.0, eye_ratio=0.0, mouth_ratio=0.0),
        ]
    else:
        raise ValueError(f"Unknown preset: {preset}")

    click.echo(f"Generating animation: preset={preset}, fps={fps}, format={fmt}")
    frames = pipeline.generate_animation_frames(img_a, img_b, keyframes, fps=fps)

    if fmt == "gif":
        save_frames_as_gif(frames, output, fps=fps)
    elif fmt == "apng":
        save_frames_as_apng(frames, output, fps=fps)
    elif fmt == "mp4":
        save_frames_as_mp4(frames, output, fps=fps)
    elif fmt == "png":
        save_frames_as_png(frames, output)

    click.echo(f"Done! Animation saved to {output}")


@cli.command()
@click.option("--image", required=True, type=click.Path(exists=True), help="Input anime illustration")
@click.option("--output", "-o", required=True, help="Output visualization path")
@click.pass_context
def masks(ctx, image, output):
    """Visualize segmentation masks for an image."""
    config = ctx.obj["config"]
    pipeline = InbetweenPipeline(config)

    img = pipeline.load_image(image)
    click.echo("Computing segmentation masks...")

    seg_masks = pipeline.segmenter.segment(img)

    # Create visualization
    h, w = img.shape[:2]
    colors = {
        "eye": (0, 255, 0),       # Green
        "mouth": (0, 0, 255),     # Red
        "face": (255, 200, 0),    # Cyan-ish
        "hair": (200, 0, 200),    # Purple
        "skin": (0, 200, 200),    # Yellow-ish
        "clothes": (200, 200, 0), # Teal
    }

    overlay = img.copy()
    for class_name, color in colors.items():
        if class_name in seg_masks:
            mask = seg_masks[class_name]
            overlay[mask > 127] = (
                overlay[mask > 127].astype(np.float32) * 0.5
                + np.array(color, dtype=np.float32) * 0.5
            ).astype(np.uint8)

    # Create side-by-side comparison
    comparison = np.hstack([img, overlay])

    # Add individual mask panels
    individual_panels = []
    for class_name in ["eye", "mouth"]:
        if class_name in seg_masks:
            mask_vis = cv2.cvtColor(seg_masks[class_name], cv2.COLOR_GRAY2BGR)
            # Add label
            cv2.putText(mask_vis, class_name, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            individual_panels.append(mask_vis)

    if individual_panels:
        masks_row = np.hstack(individual_panels)
        # Resize masks row to match comparison width
        target_w = comparison.shape[1]
        scale = target_w / masks_row.shape[1]
        masks_row = cv2.resize(masks_row, (target_w, int(masks_row.shape[0] * scale)))
        comparison = np.vstack([comparison, masks_row])

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output, comparison)
    click.echo(f"Saved mask visualization: {output}")


if __name__ == "__main__":
    cli()
