import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
	project_root = Path(__file__).resolve().parent.parent
	parser = argparse.ArgumentParser(description="Export Ultralytics YOLO .pt weights to ONNX")
	parser.add_argument("--weights", type=str, default=str(project_root / "runs" / "yolo-nano-ball-optim" / "weights" / "best.pt"), help="Path to .pt weights")
	parser.add_argument("--imgsz", type=int, default=640, help="Export image size (square)")
	parser.add_argument("--opset", type=int, default=12, help="ONNX opset version (12 for broad ARM compatibility)")
	parser.add_argument("--half", action="store_true", help="Export FP16 (if supported)")
	parser.add_argument("--dynamic", action="store_true", help="Enable dynamic axes for variable resolution")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	model = YOLO(args.weights)
	out = model.export(
		format="onnx",
		imgsz=args.imgsz,
		opset=args.opset,
		half=args.half,
		dynamic=args.dynamic,
		verbose=True,
	)
	print(f"Exported ONNX: {out}")


if __name__ == "__main__":
	main()



