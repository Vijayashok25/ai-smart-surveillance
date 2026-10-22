"""CLI Entry point for the AI Smart Surveillance System."""

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import cv2

from src.pipeline import SurveillancePipeline
from src.utils import load_config, setup_logger


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Smart Surveillance System - Real-time Multi-Object Detection, Tracking & Suspicious Behavior Alerting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        default=None,
        help="Input source: camera index (e.g. 0), video file path, or RTSP URL",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying GUI window (ideal for servers and benchmarks)",
    )
    parser.add_argument(
        "--save-output",
        "-o",
        type=str,
        default=None,
        help="Path to save annotated video output (e.g. output.mp4)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Override YOLOv8 model file or name (e.g. yolov8n.pt, yolov8s.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Override detection confidence threshold (e.g. 0.5)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help="Override NMS IoU threshold (e.g. 0.45)",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default=None,
        help="Override compute device: 'auto', 'cpu', 'cuda', 'mps'",
    )
    return parser.parse_args()


def main() -> None:
    """Main CLI execution loop."""
    args = parse_args()
    logger = setup_logger("CLI")

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Apply CLI argument overrides
    if args.model:
        config.setdefault("detection", {})["model_path"] = args.model
    if args.conf is not None:
        config.setdefault("detection", {})["confidence_threshold"] = args.conf
    if args.iou is not None:
        config.setdefault("detection", {})["iou_threshold"] = args.iou
    if args.device:
        config.setdefault("system", {})["device"] = args.device

    # Initialize master pipeline
    logger.info("Initializing AI Smart Surveillance Pipeline...")
    pipeline = SurveillancePipeline(
        config=config,
        override_source=args.source,
        logger=logger,
    )

    # Setup video writer if save-output specified
    writer: Optional[cv2.VideoWriter] = None
    save_path = Path(args.save_output) if args.save_output else None
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    # Graceful termination handler
    def handle_signal(sig, frame):
        logger.info("\nReceived stop signal. Shutting down gracefully...")
        pipeline.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    window_name = "AI Smart Surveillance System - Live Feed"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    total_frames = 0
    start_time = time.perf_counter()

    logger.info("Starting surveillance stream. Press 'q' or ESC in display window, or Ctrl+C to exit.")

    try:
        for annotated_frame, tracks, alerts, fps, inf_ms in pipeline.stream_frames():
            total_frames += 1

            # Initialize video writer on first frame
            if save_path and writer is None:
                h, w = annotated_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(save_path), fourcc, 20.0, (w, h))
                logger.info(f"Recording output to: {save_path} ({w}x{h} @ 20fps)")

            if writer is not None:
                writer.write(annotated_frame)

            # Display window
            if not args.headless:
                cv2.imshow(window_name, annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord("q"), 27]:  # 'q' or ESC
                    logger.info("Exit key pressed by user.")
                    break

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user.")
    finally:
        pipeline.stop()
        if writer is not None:
            writer.release()
            logger.info(f"Saved annotated video to: {save_path.resolve()}")
        if not args.headless:
            cv2.destroyAllWindows()

        elapsed = time.perf_counter() - start_time
        avg_fps = total_frames / elapsed if elapsed > 0 else 0.0

        logger.info("================ Surveillance Summary ================")
        logger.info(f"Total Frames Processed : {total_frames}")
        logger.info(f"Total Elapsed Time     : {elapsed:.2f} s")
        logger.info(f"Average Pipeline FPS   : {avg_fps:.2f} FPS")
        logger.info(f"Total Alerts Logged    : {pipeline._total_alerts_count}")
        logger.info(f"Database Location      : {config.get('database', {}).get('db_path')}")
        logger.info("======================================================")


if __name__ == "__main__":
    main()
