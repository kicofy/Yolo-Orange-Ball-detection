from pathlib import Path
import shutil
import cv2
import numpy as np
import random


def convert_split(src_images: Path, src_labels: Path, dst_images: Path, dst_labels: Path) -> int:
	dst_images.mkdir(parents=True, exist_ok=True)
	dst_labels.mkdir(parents=True, exist_ok=True)
	count = 0
	# Accept any file; robust decode even if extension is unusual (e.g., sample.jpg.XXXX)
	for img_path in sorted(src_images.glob("*")):
		if not img_path.is_file():
			continue
		img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
		if img is None:
			# try content-based decode
			try:
				data = np.fromfile(str(img_path), dtype=np.uint8)
				img = cv2.imdecode(data, cv2.IMREAD_COLOR)
			except Exception:
				img = None
		if img is None:
			continue
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
		cv2.imwrite(str(dst_images / img_path.name), img3)
		label_src = src_labels / f"{img_path.stem}.txt"
		if label_src.exists():
			shutil.copy2(label_src, dst_labels / label_src.name)
		count += 1
	return count


def write_data_yaml(dst_root: Path, out_yaml: Path, use_split: bool) -> None:
	if use_split:
		train = dst_root / "images" / "train"
		val = dst_root / "images" / "val"
		content = (
			f"train: \"{train}\"\n"
			f"val: \"{val}\"\n"
			"names:\n  0: ball\n"
		)
	else:
		images = dst_root / "images"
		# 当没有拆分时，train/val 都指向同一目录（保持原结构不变）
		content = (
			f"train: \"{images}\"\n"
			f"val: \"{images}\"\n"
			"names:\n  0: ball\n"
		)
	out_yaml.write_text(content, encoding="utf-8")


def main() -> None:
	project_root = Path(__file__).resolve().parent.parent
	src_root = project_root / "yolo-data"
	dst_root = project_root / "yolo-data-gray"

	counts = {"train": 0, "val": 0}
	images_train_dir = src_root / "images" / "train"
	images_val_dir = src_root / "images" / "val"
	labels_train_dir = src_root / "labels" / "train"
	labels_val_dir = src_root / "labels" / "val"

	use_split = images_train_dir.exists() or images_val_dir.exists()
	if use_split:
		# Standard YOLO split present
		counts["train"] = convert_split(
			images_train_dir,
			labels_train_dir if labels_train_dir.exists() else (src_root / "labels" / "train"),
			dst_root / "images" / "train",
			dst_root / "labels" / "train",
		)
		counts["val"] = convert_split(
			images_val_dir,
			labels_val_dir if labels_val_dir.exists() else (src_root / "labels" / "val"),
			dst_root / "images" / "val",
			dst_root / "labels" / "val",
		)
	else:
		# Preserve flat structure: images/ and labels/ only
		src_images_flat = src_root / "images"
		src_labels_flat = src_root / "labels"
		dst_images_flat = dst_root / "images"
		dst_labels_flat = dst_root / "labels"
		dst_images_flat.mkdir(parents=True, exist_ok=True)
		dst_labels_flat.mkdir(parents=True, exist_ok=True)
		c = 0
		for img_path in sorted(src_images_flat.glob("*")):
			if not img_path.is_file():
				continue
			img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
			if img is None:
				try:
					data = np.fromfile(str(img_path), dtype=np.uint8)
					img = cv2.imdecode(data, cv2.IMREAD_COLOR)
				except Exception:
					img = None
			if img is None:
				continue
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			img3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
			cv2.imwrite(str(dst_images_flat / img_path.name), img3)
			label_src = src_labels_flat / f"{img_path.stem}.txt"
			if label_src.exists():
				shutil.copy2(label_src, dst_labels_flat / label_src.name)
			c += 1
		counts["train"] = c
		counts["val"] = 0

	out_yaml = project_root / "yolo-data" / "data_gray.yaml"
	write_data_yaml(dst_root, out_yaml, use_split)

	print(f"Gray dataset written to: {dst_root}")
	print(f"Train images: {counts['train']}, Val images: {counts['val']}")
	print(f"Data YAML: {out_yaml}")


if __name__ == "__main__":
	main()



