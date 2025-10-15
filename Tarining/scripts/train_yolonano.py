from pathlib import Path

from ultralytics import YOLO


def main() -> None:
	project_root = Path(__file__).resolve().parent.parent
	data_yaml = project_root / "yolo-data" / "data.yaml"

	# Use a stronger base model for best accuracy on small dataset
	model = YOLO("yolov8s.pt")

	# Train (optimized for accuracy)
	results = model.train(
		data=str(data_yaml),
		epochs=10000,
		imgsz=960,
		batch=8,
		patience=100,
		single_cls=True,
		optimizer="AdamW",
		lr0=0.002,
		cos_lr=True,
		multi_scale=True,
		cache=True,
		device="mps",
		workers=0,
		project=str(project_root / "runs"),
		name="yolo-nano-ball-optim",
		verbose=True,
	)

	print(results)


if __name__ == "__main__":
	main()


