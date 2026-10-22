"""Utility to download a free public sample surveillance video or generate a synthetic benchmark video."""

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests


def download_sample_video(target_path: Path, url: Optional[str] = None) -> bool:
    """Attempts to download a sample video from a public repository."""
    if url is None:
        # High quality royalty-free CCTV surveillance sample
        url = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4"

    target_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading sample video from: {url}")
    try:
        response = requests.get(url, stream=True, timeout=15.0)
        if response.status_code == 200:
            with open(target_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    f.write(chunk)
            print(f"Successfully downloaded sample video to: {target_path} ({target_path.stat().st_size / 1024:.1f} KB)")
            return True
        else:
            print(f"Download returned HTTP {response.status_code}.")
            return False
    except Exception as e:
        print(f"Download failed ({e}). Proceeding to synthetic generation...")
        return False


def generate_synthetic_surveillance_video(
    target_path: Path,
    duration_sec: int = 15,
    fps: int = 25,
    width: int = 640,
    height: int = 480,
) -> Path:
    """Generates a realistic surveillance scene with animated persons, bags, and zone intrusion.

    Guarantees reliable zero-network offline testing and benchmarking.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(target_path), fourcc, float(fps), (width, height))

    print(f"Generating synthetic surveillance benchmark video: {target_path} ({total_frames} frames)...")

    # Restricted zone polygon: [(150, 150), (500, 150), (500, 450), (150, 450)]
    zone_poly = np.array([[150, 150], [500, 150], [500, 450], [150, 450]], np.int32)

    for i in range(total_frames):
        # Background: Security surveillance floor and wall
        frame = np.ones((height, width, 3), dtype=np.uint8) * 45
        # Floor
        cv2.rectangle(frame, (0, 120), (width, height), (75, 75, 75), -1)
        # Floor tiles / grid lines
        for y in range(120, height, 60):
            cv2.line(frame, (0, y), (width, y), (65, 65, 65), 1)
        for x in range(0, width, 80):
            cv2.line(frame, (x, 120), (x, height), (65, 65, 65), 1)

        # Draw monitored zone outline on floor
        cv2.polylines(frame, [zone_poly], isClosed=True, color=(30, 30, 180), thickness=2)

        # -------------------------------------------------------------
        # Entity 1: Person walking into zone, loitering, and leaving bag
        # -------------------------------------------------------------
        # Movement trajectory:
        # Frames 0-75 (0-3s): Walks from (50, 200) to (250, 250) (inside zone)
        # Frames 76-250 (3-10s): Loiters stationary at (250, 250)
        # Frames 251-375 (10-15s): Walks away to (600, 200)
        if i <= 75:
            progress = i / 75.0
            px = int(50 + progress * 200)
            py = int(200 + progress * 50)
        elif i <= 250:
            px = 250
            py = 250
        else:
            progress = (i - 250) / 125.0
            px = int(250 + progress * 350)
            py = int(250 - progress * 50)

        # Draw Person (Humanoid silhouette: head, body, legs)
        # Head
        cv2.circle(frame, (px, py - 60), 14, (180, 150, 120), -1)
        # Body / Jacket
        cv2.rectangle(frame, (px - 18, py - 46), (px + 18, py), (40, 60, 140), -1)
        # Legs
        leg_offset = int(np.sin(i * 0.4) * 6) if (i <= 75 or i > 250) else 0
        cv2.line(frame, (px - 8, py), (px - 8 + leg_offset, py + 40), (30, 30, 30), 6)
        cv2.line(frame, (px + 8, py), (px + 8 - leg_offset, py + 40), (30, 30, 30), 6)

        # -------------------------------------------------------------
        # Entity 2: Bag / Backpack left behind at frame 150 (stationary)
        # -------------------------------------------------------------
        if i >= 100:
            bag_x, bag_y = 230, 275
            # Draw Backpack / Luggage shape
            cv2.rectangle(frame, (bag_x - 14, bag_y - 18), (bag_x + 14, bag_y + 10), (120, 80, 40), -1)
            cv2.circle(frame, (bag_x, bag_y - 18), 8, (140, 100, 50), -1)
            cv2.rectangle(frame, (bag_x - 8, bag_y - 10), (bag_x + 8, bag_y + 6), (90, 60, 30), -1)

        # -------------------------------------------------------------
        # Entity 3: Second person entering at frame 200 (Crowding)
        # -------------------------------------------------------------
        if i >= 180:
            p2_x = int(120 + (i - 180) * 1.5)
            p2_y = 320
            if p2_x < width:
                cv2.circle(frame, (p2_x, p2_y - 60), 14, (160, 140, 110), -1)
                cv2.rectangle(frame, (p2_x - 16, p2_y - 46), (p2_x + 16, p2_y), (80, 120, 50), -1)
                leg2_offset = int(np.sin(i * 0.4) * 5)
                cv2.line(frame, (p2_x - 6, p2_y), (p2_x - 6 + leg2_offset, p2_y + 38), (30, 30, 30), 5)
                cv2.line(frame, (p2_x + 6, p2_y), (p2_x + 6 - leg2_offset, p2_y + 38), (30, 30, 30), 5)

        # Timestamp watermark
        time_text = f"CAM-01 2026-10-21 23:45:{i // fps:02d}.{int((i % fps) / fps * 100):02d}"
        cv2.putText(frame, time_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

        out.write(frame)

    out.release()
    print(f"Generated benchmark video: {target_path} ({duration_sec}s @ {fps} FPS)")
    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare surveillance sample video for testing and evaluation")
    parser.add_argument("--output", type=str, default="samples/sample_surveillance.mp4", help="Target output file path")
    parser.add_argument("--download", action="store_true", help="Attempt downloading public video from web")
    args = parser.parse_args()

    out_path = Path(args.output)
    if args.download:
        success = download_sample_video(out_path)
        if success:
            return

    # Fallback / Default: generate synthetic video
    generate_synthetic_surveillance_video(out_path)


if __name__ == "__main__":
    main()
