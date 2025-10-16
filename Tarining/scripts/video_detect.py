import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
	project_root = Path(__file__).resolve().parent
	default_weights = project_root.parent / "runs" / "yolo-nano-ball-optim1" / "weights" / "best.pt"
	parser = argparse.ArgumentParser(description="Run YOLO detection on a video and display with OpenCV")
	parser.add_argument("--source", type=str, required=True, help="Path to input video file")
	parser.add_argument("--weights", type=str, default=str(default_weights))
	parser.add_argument("--imgsz", type=int, default=640)
	parser.add_argument("--conf", type=float, default=0.25)
	parser.add_argument("--device", type=str, default="cpu")
	parser.add_argument("--save", action="store_true", help="Save annotated video")
	parser.add_argument("--out", type=str, default="out.mp4", help="Output video path when --save is set")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	model = YOLO(args.weights)
	cap = cv2.VideoCapture(args.source)
	if not cap.isOpened():
		print(f"Failed to open video: {args.source}")
		return

	fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
	w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	writer = None
	if args.save:
		fourcc = cv2.VideoWriter_fourcc(*"mp4v")
		writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))

	prev_t = time.time()
	while True:
		ok, frame = cap.read()
		if not ok:
			break
		res = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
		plot = res[0].plot()
		now = time.time()
		cur_fps = 1.0 / max(1e-3, now - prev_t)
		prev_t = now
		cv2.putText(plot, f"FPS: {cur_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
		if writer is not None:
			writer.write(plot)
		cv2.imshow("Video Detect", plot)
		key = cv2.waitKey(1) & 0xFF
		if key in (27, ord("q")):
			break

	cap.release()
	if writer is not None:
		writer.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()


