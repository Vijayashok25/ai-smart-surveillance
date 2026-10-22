"""Interactive OpenCV utility to define polygonal surveillance zones and update config.yaml."""

import argparse
import sys
from pathlib import Path
from typing import Any, List, Tuple

import cv2
import yaml

# Add root folder to sys.path so utils can be loaded
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, setup_logger

logger = setup_logger("ZoneSelector")

# Global click state
points: List[Tuple[int, int]] = []
current_mouse: Tuple[int, int] = (0, 0)


def mouse_callback(event: int, x: int, y: int, flags: int, param: Any) -> None:
    """Handles mouse clicks and moves for polygon drawing."""
    global points, current_mouse
    current_mouse = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        logger.info(f"Added vertex #{len(points)}: ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            removed = points.pop()
            logger.info(f"Removed vertex: {removed}")


def main() -> None:
    """Main execution routine for zone selector tool."""
    parser = argparse.ArgumentParser(
        description="Click vertices to define a restricted surveillance polygon and save to config.yaml"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video file, webcam index (0), or image path (defaults to config.yaml source)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml to update",
    )
    parser.add_argument(
        "--rule",
        type=str,
        default="both",
        choices=["zone_intrusion", "loitering", "both"],
        help="Which rule polygon to update in config.yaml",
    )
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.is_file():
        logger.error(f"Config file not found: {config_path}")
        return

    config = load_config(config_path)
    source = args.source if args.source is not None else config.get("video", {}).get("source", 0)

    # Convert numeric source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    # Capture first frame
    logger.info(f"Connecting to source: {source}...")
    frame = None

    if isinstance(source, str) and Path(source).suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
        frame = cv2.imread(str(source))
    else:
        if isinstance(source, int):
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(str(source))

        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()

    if frame is None:
        logger.error(f"Could not read any frame from source: {source}")
        logger.info("Tip: You can pass a sample image using --source path/to/image.jpg")
        return

    h, w = frame.shape[:2]
    logger.info(f"Frame loaded ({w}x{h}). Instructions:")
    logger.info("  [Left Click]   : Add polygon vertex")
    logger.info("  [Right Click]  : Remove last vertex")
    logger.info("  [Key 'r']      : Reset all vertices")
    logger.info("  [Key 's']      : Save polygon to config.yaml and exit")
    logger.info("  [Key 'q' / ESC]: Quit without saving")

    window_name = "AI Smart Surveillance - Define Polygon Zone"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        display = frame.copy()

        # Draw existing polygon edges
        if len(points) > 0:
            for i in range(len(points)):
                cv2.circle(display, points[i], 5, (0, 255, 0), -1)
                cv2.putText(
                    display,
                    str(i + 1),
                    (points[i][0] + 8, points[i][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            if len(points) > 1:
                for i in range(1, len(points)):
                    cv2.line(display, points[i - 1], points[i], (0, 255, 255), 2)

            # Draw preview edge to current cursor
            cv2.line(display, points[-1], current_mouse, (255, 255, 0), 1, cv2.LINE_AA)

            # Close preview if 3+ points
            if len(points) >= 3:
                cv2.line(display, points[-1], points[0], (0, 0, 255), 1, cv2.LINE_AA)

        # Instructions banner
        instruction_text = (
            f"Vertices: {len(points)} | [Left Click] Add | [Right] Undo | [r] Reset | [s] Save | [q] Quit"
        )
        cv2.rectangle(display, (0, 0), (w, 30), (30, 30, 30), -1)
        cv2.putText(
            display,
            instruction_text,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF

        if key in [ord("q"), 27]:  # ESC or q
            logger.info("Exited without saving.")
            break

        elif key == ord("r"):
            points.clear()
            logger.info("Cleared all points.")

        elif key == ord("s"):
            if len(points) < 3:
                logger.warning("A valid polygon requires at least 3 vertices!")
                continue

            polygon_data = [[int(pt[0]), int(pt[1])] for pt in points]
            logger.info(f"Saving polygon: {polygon_data}")

            # Update configuration
            if args.rule in ["zone_intrusion", "both"]:
                if "zone_intrusion" in config.get("rules", {}):
                    config["rules"]["zone_intrusion"]["polygon"] = polygon_data

            if args.rule in ["loitering", "both"]:
                if "loitering" in config.get("rules", {}):
                    config["rules"]["loitering"]["polygon"] = polygon_data

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, default_flow_style=None, sort_keys=False)

            logger.info(f"Successfully updated polygon in {config_path.resolve()}!")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
