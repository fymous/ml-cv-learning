"""Lab worker entrypoint.

Examples:
  python -m app.main --source ../testdata/sample.mp4
  python -m app.main --source ../testdata/sample.mp4 --zones ../testdata/zones.yaml
  python -m app.main --source 0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.camera_loop import CameraLoop, CameraSpec
from app.zone_loader import load_zones

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("cv-lab")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local CV lab worker")
    parser.add_argument("--source", required=True, help="Video file, RTSP URL, or webcam index")
    parser.add_argument("--zones", default="", help="Optional zones YAML")
    parser.add_argument("--camera-id", default="lab")
    args = parser.parse_args()

    zones = []
    if args.zones:
        zones = load_zones(args.zones)
        logger.info("loaded %d zones from %s", len(zones), args.zones)

    spec = CameraSpec(camera_id=args.camera_id, source=args.source, zones=zones)
    CameraLoop(spec).run()


if __name__ == "__main__":
    # Allow `python app/main.py` from worker/
    import sys

    worker_root = Path(__file__).resolve().parents[1]
    if str(worker_root) not in sys.path:
        sys.path.insert(0, str(worker_root))
    main()
