import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
	project_root = Path(__file__).resolve().parent
	default_weights = project_root.parent / "runs" / "yolo-nano-ball-optim" / "weights" / "best.pt"
	parser = argparse.ArgumentParser(description="Realtime USB camera detection with frame drop if busy")
	parser.add_argument("--cam", type=int, default=0, help="Camera index, e.g., 0 for /dev/video0")
	parser.add_argument("--weights", type=str, default=str(default_weights))
	parser.add_argument("--imgsz", type=int, default=320)
	parser.add_argument("--conf", type=float, default=0.25)
	parser.add_argument("--device", type=str, default="cpu")
	parser.add_argument("--width", type=int, default=1280)
	parser.add_argument("--height", type=int, default=800)
	parser.add_argument("--fps", type=int, default=120)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	# Open camera with V4L2 and MJPG to reduce CPU decode load
	cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
	cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
	cap.set(cv2.CAP_PROP_FPS, args.fps)
	if not cap.isOpened():
		print(f"Failed to open camera index {args.cam}")
		return

	model = YOLO(args.weights)
	executor = ThreadPoolExecutor(max_workers=1)
	inflight = None
	last_plot = None
	prev_t = time.time()

	def infer(img):
		res = model.predict(source=img, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
		return res[0].plot()

	while True:
		ok, frame = cap.read()
		if not ok:
			break

		# If inference worker is idle, submit current frame; else drop it (no queue buildup)
		if inflight is None or inflight.done():
			if inflight is not None:
				try:
					last_plot = inflight.result(timeout=0)
				except Exception:
					last_plot = None
			inflight = executor.submit(infer, frame.copy())

		# Render: prefer latest plotted result if available, else raw frame
		now = time.time()
		fps = 1.0 / max(1e-3, now - prev_t)
		prev_t = now
		canvas = last_plot if last_plot is not None else frame
		cv2.putText(canvas, f"FPS:{fps:.1f} (imgsz {args.imgsz})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
		cv2.imshow("cam_detect", canvas)
		if cv2.waitKey(1) & 0xFF == 27:
			break

	cap.release()
	executor.shutdown(wait=False, cancel_futures=True)
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()


