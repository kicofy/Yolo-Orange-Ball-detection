from pathlib import Path
import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Train YOLO for orange-ball detection (supports grayscale)")
	project_root = Path(__file__).resolve().parent.parent
	p.add_argument("--data", type=str, default=str(project_root / "yolo-data" / "data.yaml"))
	p.add_argument("--model", type=str, default="yolov8n.pt")
	p.add_argument("--epochs", type=int, default=300)
	p.add_argument("--imgsz", type=int, default=640)
	p.add_argument("--batch", type=int, default=8)
	p.add_argument("--device", type=str, default="cpu")
	p.add_argument("--name", type=str, default="yolo-nano-ball-optim")
	p.add_argument("--gray", action="store_true", help="Use grayscale dataset (data_gray.yaml) and disable color aug")
	return p.parse_args()


def main() -> None:
	args = parse_args()
	project_root = Path(__file__).resolve().parent.parent
	data_yaml = Path(args.data)
	if args.gray:
		cand = project_root / "yolo-data" / "data_gray.yaml"
		if cand.exists():
			data_yaml = cand

	model = YOLO(args.model)

	train_kwargs = dict(
		data=str(data_yaml),
		epochs=args.epochs,
		imgsz=args.imgsz,
		batch=args.batch,
		patience=100,
		single_cls=True,
		optimizer="AdamW",
		lr0=0.002,
		cos_lr=True,
		multi_scale=True,
		cache=True,
		device=args.device,
		workers=0,
		project=str(project_root / "runs"),
		name=args.name,
		verbose=True,
	)
	if args.gray:
		# Stable settings for grayscale small dataset
		train_kwargs.update(
			dict(
				hsv_h=0.0,
				hsv_s=0.0,
				multi_scale=False,
				lr0=0.001,
				amp=False,
			)
		)

	results = model.train(**train_kwargs)
	print(results)


if __name__ == "__main__":
	main()


