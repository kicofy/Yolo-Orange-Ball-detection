import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class InferenceConfig:
	model_path: str
	imgsz: int
	conf_threshold: float
	iou_threshold: float
	providers: List[str]
	threads: int
	debug: bool


def parse_args() -> argparse.Namespace:
	project_root = Path(__file__).resolve().parent.parent
	parser = argparse.ArgumentParser(description="ONNX Runtime video inference with OpenCV display")
	parser.add_argument("--onnx", type=str, default=str(project_root / "runs" / "yolo-nano-ball-optim" / "weights" / "best.onnx"))
	parser.add_argument("--source", type=str, default=str(project_root / "Dataset" / "video" / "06e36a63e4befd4c8ecace67178b66ca.mp4"), help="Video source: file path or webcam index like 0")
	parser.add_argument("--imgsz", type=int, default=640)
	parser.add_argument("--conf", type=float, default=0.25)
	parser.add_argument("--iou", type=float, default=0.45)
	parser.add_argument("--providers", type=str, nargs="*", default=["CPUExecutionProvider"], help="ONNX Runtime providers")
	parser.add_argument("--threads", type=int, default=4, help="Intra-op threads for ORT")
	parser.add_argument("--debug", action="store_true", help="Show extra diagnostics overlays")
	return parser.parse_args()


def letterbox(im: np.ndarray, new_size: int = 640, color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[float, float]]:
	h, w = im.shape[:2]
	r = min(new_size / h, new_size / w)
	new_unpad = (int(round(w * r)), int(round(h * r)))
	# padding as floats for accurate reverse mapping
	dw = (new_size - new_unpad[0]) / 2.0
	dh = (new_size - new_unpad[1]) / 2.0
	resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
	canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
	ih, iw = int(round(dh)), int(round(dw))
	canvas[ih:ih + new_unpad[1], iw:iw + new_unpad[0]] = resized
	return canvas, r, (dw, dh)


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
	y = x.copy()
	y[:, 0] = x[:, 0] - x[:, 2] / 2
	y[:, 1] = x[:, 1] - x[:, 3] / 2
	y[:, 2] = x[:, 0] + x[:, 2] / 2
	y[:, 3] = x[:, 1] + x[:, 3] / 2
	return y


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
	if boxes.size == 0:
		return []
	idxs = scores.argsort()[::-1]
	selected: List[int] = []
	while idxs.size > 0:
		current = int(idxs[0])
		selected.append(current)
		if idxs.size == 1:
			break
		rest = idxs[1:]
		x1 = np.maximum(boxes[current, 0], boxes[rest, 0])
		y1 = np.maximum(boxes[current, 1], boxes[rest, 1])
		x2 = np.minimum(boxes[current, 2], boxes[rest, 2])
		y2 = np.minimum(boxes[current, 3], boxes[rest, 3])
		w = np.maximum(0, x2 - x1)
		h = np.maximum(0, y2 - y1)
		inter = w * h
		area_current = (boxes[current, 2] - boxes[current, 0]) * (boxes[current, 3] - boxes[current, 1])
		area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
		iou = inter / (area_current + area_rest - inter + 1e-7)
		idxs = rest[iou <= iou_thres]
	return selected


def prepare_session(cfg: InferenceConfig) -> ort.InferenceSession:
	ops = ort.SessionOptions()
	ops.intra_op_num_threads = cfg.threads
	return ort.InferenceSession(cfg.model_path, sess_options=ops, providers=cfg.providers)


def parse_outputs_to_boxes_scores(pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Convert various ONNX head/NMS outputs to (boxes_xyxy, scores, cls_ids) in letterbox space.
	Supports:
	- Built-in NMS: [N, 6] => x1,y1,x2,y2,score,cls
	- Single-class raw heads: [N,5] (cx,cy,w,h,obj) or [N,6] with cls0
	- Multi-class raw heads: [N,no] with obj + class probs
	"""
	N, no = pred.shape[0], pred.shape[1]
	if no == 6 and N <= 300:
		# NMS output
		boxes = pred[:, 0:4]
		scores = pred[:, 4]
		cls_ids = pred[:, 5].astype(np.int32)
		return boxes, scores, cls_ids

	if no == 5:
		# single-class raw without class logit
		boxes = xywh2xyxy(pred[:, 0:4])
		scores = pred[:, 4]
		cls_ids = np.zeros((N,), dtype=np.int32)
		return boxes, scores, cls_ids

	if no == 6:
		# single-class raw with one class logit
		boxes = xywh2xyxy(pred[:, 0:4])
		scores = pred[:, 4] * pred[:, 5]
		cls_ids = np.zeros((N,), dtype=np.int32)
		return boxes, scores, cls_ids

	# multi-class raw: cx,cy,w,h,obj,cls...
	boxes = xywh2xyxy(pred[:, 0:4])
	objectness = pred[:, 4:5]
	class_scores = pred[:, 5:]
	if class_scores.size == 0:
		scores = objectness.squeeze(1)
		cls_ids = np.zeros((N,), dtype=np.int32)
	else:
		best_cls = class_scores.argmax(axis=1)
		scores = (objectness * class_scores[np.arange(N), best_cls][:, None]).squeeze(1)
		cls_ids = best_cls.astype(np.int32)
	return boxes, scores, cls_ids


def run_inference(sess: ort.InferenceSession, image_bgr: np.ndarray, imgsz: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[float, float, float], np.ndarray]:
	img, r, (dw, dh) = letterbox(image_bgr, imgsz)
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img_chw = np.transpose(img_rgb, (2, 0, 1))[None].astype(np.float32)
	img_chw = np.ascontiguousarray(img_chw / 255.0)

	input_name = sess.get_inputs()[0].name
	outputs = sess.run(None, {input_name: img_chw})
	# possible multiple outputs for some exports: handle first with largest size
	pred = outputs[0]
	if pred.ndim == 3:
		# [B, *, *] -> ensure [B, N, no]
		if pred.shape[1] < pred.shape[2]:
			pred = np.transpose(pred, (0, 2, 1))
		pred = pred[0]
	elif pred.ndim == 2:
		# already [N, no]
		pass
	else:
		raise RuntimeError(f"Unexpected output rank: {pred.ndim}")

	boxes_l, scores, cls_ids = parse_outputs_to_boxes_scores(pred)
	boxes_l = boxes_l.astype(np.float32)

	# Normalize handling: if values look like [0,1], scale to imgsz first
	max_coord = float(boxes_l.max()) if boxes_l.size else 0.0
	min_coord = float(boxes_l.min()) if boxes_l.size else 0.0
	if 0.0 <= min_coord and max_coord <= 1.01:
		boxes_l *= float(imgsz)

	# Map from letterbox to original image size
	boxes = boxes_l.copy()
	if boxes.size:
		boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / max(r, 1e-6)
		boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / max(r, 1e-6)

	return boxes, scores, cls_ids, (r, dw, dh), pred


def main() -> None:
	args = parse_args()
	cfg = InferenceConfig(
		model_path=args.onnx,
		imgsz=args.imgsz,
		conf_threshold=args.conf,
		iou_threshold=args.iou,
		providers=args.providers,
		threads=args.threads,
		debug=bool(args.debug),
	)

	sess = prepare_session(cfg)

	# Open source
	if args.source.isdigit():
		source = int(args.source)
	else:
		source = args.source
	cap = cv2.VideoCapture(source)
	if not cap.isOpened():
		print(f"Failed to open source: {args.source}")
		return

	prev = time.time()
	while True:
		ok, frame = cap.read()
		if not ok:
			break
		boxes, scores, cls_ids, (r, dw, dh), pred = run_inference(sess, frame, cfg.imgsz)

		# Filter by confidence
		mask = scores >= cfg.conf_threshold
		boxes = boxes[mask]
		scores_f = scores[mask]
		cls_ids_f = cls_ids[mask]

		# NMS
		pre_n = int(boxes.shape[0])
		if boxes.size > 0:
			keep = nms(boxes, scores_f, cfg.iou_threshold)
			boxes = boxes[keep]
			scores_f = scores_f[keep]
			cls_ids_f = cls_ids_f[keep]
		post_n = int(boxes.shape[0])

		# Clip and order
		if boxes.size > 0:
			H, W = frame.shape[:2]
			boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, W - 1)
			boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, H - 1)
			x1 = np.minimum(boxes[:, 0], boxes[:, 2])
			y1 = np.minimum(boxes[:, 1], boxes[:, 3])
			x2 = np.maximum(boxes[:, 0], boxes[:, 2])
			y2 = np.maximum(boxes[:, 1], boxes[:, 3])
			boxes = np.stack([x1, y1, x2, y2], axis=1)

		# Draw
		for (x1, y1, x2, y2), s, c in zip(boxes.astype(int), scores_f, cls_ids_f):
			cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
			cv2.putText(frame, f"ball {s:.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

		now = time.time()
		fps = 1.0 / max(1e-3, now - prev)
		prev = now
		cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
		cv2.putText(frame, f"Det: {pre_n}->{post_n}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
		if boxes.size > 0:
			x1, y1, x2, y2 = boxes[0].astype(int)
			cv2.putText(frame, f"B0: {x1},{y1},{x2},{y2}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
		if cfg.debug:
			cv2.putText(frame, f"r={r:.3f} dw={dw:.1f} dh={dh:.1f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
			try:
				show = pred[:1, :min(8, pred.shape[1])]
				cv2.putText(frame, f"raw0: {np.array2string(show[0], precision=2)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
			except Exception:
				pass

		cv2.imshow("ONNX Inference", frame)
		if cv2.getWindowProperty("ONNX Inference", cv2.WND_PROP_VISIBLE) < 1:
			break
		if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
			break

	cap.release()
	cv2.destroyAllWindows()
	cv2.waitKey(1)


if __name__ == "__main__":
	main()



