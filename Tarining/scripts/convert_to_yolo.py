import json
import shutil
from pathlib import Path

from PIL import Image


def load_bboxes(labels_path: Path) -> dict[str, list[dict]]:
	with labels_path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	# Expected schema: {"version": 1, "type": "bounding-box-labels", "boundingBoxes": { filename: [ {label,x,y,width,height}, ... ] } }
	return data.get("boundingBoxes", {})


def convert_box_to_yolo(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> tuple[float, float, float, float]:
	# Input x,y are top-left pixel coordinates; w,h are pixel sizes.
	cx = (x + w / 2.0) / float(img_w)
	cy = (y + h / 2.0) / float(img_h)
	nw = w / float(img_w)
	nh = h / float(img_h)
	# Clamp to [0,1] to avoid numeric drift
	cx = min(max(cx, 0.0), 1.0)
	cy = min(max(cy, 0.0), 1.0)
	nw = min(max(nw, 0.0), 1.0)
	nh = min(max(nh, 0.0), 1.0)
	return cx, cy, nw, nh


def write_yolo_label(txt_path: Path, boxes: list[tuple[int, float, float, float, float]]) -> None:
	lines = [f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cls_idx, cx, cy, w, h in boxes]
	txt_path.parent.mkdir(parents=True, exist_ok=True)
	txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def process_split(src_dir: Path, labels_file: Path, images_out: Path, labels_out: Path, class_to_index: dict[str, int]) -> int:
	bboxes_map = load_bboxes(labels_file)
	count_images = 0
	for filename, anns in bboxes_map.items():
		src_img = src_dir / filename
		if not src_img.exists():
			# Try to find the image by name only (in case paths differ)
			candidates = list(src_dir.glob(filename))
			if candidates:
				src_img = candidates[0]
			else:
				# Skip missing image
				continue

		# Read image size
		with Image.open(src_img) as im:
			img_w, img_h = im.size

		# Convert all boxes for this image
		boxes_yolo: list[tuple[int, float, float, float, float]] = []
		for ann in anns:
			label = ann.get("label", "ball")
			if label not in class_to_index:
				# Unknown label; skip
				continue
			x = float(ann["x"]) ; y = float(ann["y"]) ; w = float(ann["width"]) ; h = float(ann["height"]) 
			cx, cy, nw, nh = convert_box_to_yolo(x, y, w, h, img_w, img_h)
			boxes_yolo.append((class_to_index[label], cx, cy, nw, nh))

		# Copy image
		images_out.mkdir(parents=True, exist_ok=True)
		dst_img = images_out / src_img.name
		shutil.copy2(src_img, dst_img)

		# Write label file
		label_file = labels_out / (src_img.stem + ".txt")
		write_yolo_label(label_file, boxes_yolo)

		count_images += 1

	return count_images


def main() -> None:
	project_root = Path(__file__).resolve().parent.parent
	dataset_dir = project_root / "Dataset"
	train_dir = dataset_dir / "training"
	val_dir = dataset_dir / "testing"

	# Output structure compatible with Ultralytics
	out_root = project_root / "yolo-data"
	images_train = out_root / "images" / "train"
	images_val = out_root / "images" / "val"
	labels_train = out_root / "labels" / "train"
	labels_val = out_root / "labels" / "val"

	class_to_index = {"ball": 0}

	train_labels_file = train_dir / "bounding_boxes.labels"
	val_labels_file = val_dir / "bounding_boxes.labels"

	images_train.parent.mkdir(parents=True, exist_ok=True)
	labels_train.parent.mkdir(parents=True, exist_ok=True)
	images_val.parent.mkdir(parents=True, exist_ok=True)
	labels_val.parent.mkdir(parents=True, exist_ok=True)

	count_train = process_split(train_dir, train_labels_file, images_train, labels_train, class_to_index)
	count_val = process_split(val_dir, val_labels_file, images_val, labels_val, class_to_index)

	print(f"Prepared YOLO dataset at: {out_root}")
	print(f"\tTrain images: {count_train}")
	print(f"\tVal images:   {count_val}")
	print("Classes:", ", ".join(sorted(class_to_index, key=class_to_index.get)))


if __name__ == "__main__":
	main()


