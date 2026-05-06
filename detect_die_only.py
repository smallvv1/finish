#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "best_last.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runtime" / "die_only"


def _find_class_id(model: YOLO, class_name: str) -> int:
    target = class_name.strip().lower()
    names = getattr(model, "names", {}) or {}
    for class_id, name in names.items():
        if str(name).strip().lower() == target:
            return int(class_id)
    raise RuntimeError(f"Class '{class_name}' not found in model names: {names}")


def detect_die(
    image_path: Path,
    model: YOLO,
    conf: float,
    imgsz: int,
    output_dir: Path,
    max_boxes: int,
    save_image: bool = True,
) -> Dict[str, Any]:
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    die_class_id = _find_class_id(model, "die")
    results = model(img, conf=float(conf), verbose=False, imgsz=int(imgsz))
    result = results[0]

    detections: List[Dict[str, Any]] = []
    for box in result.boxes:
        class_id = int(box.cls.item()) if box.cls is not None else -1
        if class_id != die_class_id:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue

        confidence = float(box.conf.item()) if box.conf is not None else 0.0
        detections.append(
            {
                "class_id": class_id,
                "class_name": "die",
                "confidence": round(confidence, 4),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            }
        )

    detections = sorted(detections, key=lambda item: item["confidence"], reverse=True)
    if max_boxes > 0:
        detections = detections[:max_boxes]

    annotated_path = None
    if save_image:
        output_dir.mkdir(parents=True, exist_ok=True)
        annotated = img.copy()
        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            label = f"die {det['confidence']:.2f}"
            box_color = (255, 0, 0)
            text_color = (255, 255, 255)
            thickness = max(2, round(min(img.shape[:2]) / 900))
            font_scale = max(0.8, min(img.shape[:2]) / 1400)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            label_y1 = max(0, y1 - th - baseline - 4)
            label_y2 = max(th + baseline + 4, y1)
            label_x2 = min(img.shape[1], x1 + tw + 8)
            cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), box_color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 4, label_y2 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )
        annotated_path = output_dir / f"{image_path.stem}_die_only{image_path.suffix}"
        cv2.imwrite(str(annotated_path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    return {
        "image_path": str(image_path),
        "annotated_image_path": str(annotated_path) if annotated_path else None,
        "die_count": len(detections),
        "detections": detections,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect die boxes only, without OCR or missing evaluation.")
    parser.add_argument("image", help="Image path to detect.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="YOLO model path.")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference image size.")
    parser.add_argument("--max-boxes", type=int, default=5, help="Maximum die boxes to draw. Use 0 for all.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for annotated image.")
    parser.add_argument("--no-save", action="store_true", help="Do not save annotated image.")
    args = parser.parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    model = YOLO(str(model_path))
    report = detect_die(
        image_path=image_path,
        model=model,
        conf=args.conf,
        imgsz=args.imgsz,
        output_dir=Path(args.output_dir),
        max_boxes=args.max_boxes,
        save_image=not args.no_save,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
