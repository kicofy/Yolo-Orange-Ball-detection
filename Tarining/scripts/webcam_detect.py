import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
	project_root = Path(__file__).resolve().parent.parent
	parser = argparse.ArgumentParser(description="Webcam detection using trained YOLO model")
	parser.add_argument("--weights", type=str, default=str(project_root / "runs" / "yolo-nano-ball-optim" / "weights" / "best.pt"))
	parser.add_argument("--cam", type=int, default=0, help="Webcam index")
	parser.add_argument("--imgsz", type=int, default=640)
	parser.add_argument("--conf", type=float, default=0.25)
	parser.add_argument("--device", type=str, default="cpu", help="cpu or CUDA index like 0")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	model = YOLO(args.weights)
	cap = cv2.VideoCapture(args.cam)
	if not cap.isOpened():
		print(f"Failed to open webcam index {args.cam}")
		return

	prev_t = time.time()
	try:
		while True:
			ok, frame = cap.read()
			if not ok:
				break
			res = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
			plot = res[0].plot()
			now = time.time()
			fps = 1.0 / max(1e-3, now - prev_t)
			prev_t = now
			cv2.putText(plot, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
			cv2.imshow("Webcam Detect", plot)
			# Handle window close button (user clicks the X). When closed, visibility < 1.
			if cv2.getWindowProperty("Webcam Detect", cv2.WND_PROP_VISIBLE) < 1:
				break
			key = cv2.waitKey(1) & 0xFF
			if key in (27, ord("q")):
				break
	finally:
		cap.release()
		cv2.destroyAllWindows()
		# Small wait to allow the window to actually close on some backends (e.g., macOS)
		cv2.waitKey(1)


if __name__ == "__main__":
	main()


