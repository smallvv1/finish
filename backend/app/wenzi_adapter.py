#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from pathlib import Path
from typing import Dict, List, Optional

import cv2


def _extract_digits(text: str) -> str:
    if not text:
        return ""
    return "".join(re.findall(r"\d", str(text)))


def _resize_crop_for_ocr(crop, max_side: int = 640):
    h, w = crop.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return crop
    ratio = max_side / float(longest)
    new_w = max(1, int(w * ratio))
    new_h = max(1, int(h * ratio))
    return cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _extract_text_with_engine(engine, crop) -> str:
    try:
        res = list(engine.predict(crop))
    except Exception:
        return ""
    if not res:
        return ""
    first = res[0]
    if isinstance(first, dict) and first.get("rec_texts"):
        text = " ".join(first["rec_texts"])
    elif hasattr(first, "rec_texts") and getattr(first, "rec_texts"):
        text = " ".join(first.rec_texts)
    elif isinstance(first, list):
        text = " ".join([line[1][0] for line in first if len(line) > 1 and len(line[1]) > 0])
    else:
        text = ""
    return _extract_digits(text.strip())


def _get_die_class_id(model) -> Optional[int]:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        for class_id, class_name in names.items():
            if str(class_name).lower() == "die":
                return int(class_id)
    return None


def detect_and_ocr_with_wenzi(
    image_paths: List[Path],
    model,
    ocr_engine,
    conf_threshold: float,
    code_pattern: str,
    annotated_dir: Path,
    min_box_area: int = 900,
    max_die_boxes: int = 0,
) -> List[Dict]:
    reports: List[Dict] = []
    annotated_dir.mkdir(parents=True, exist_ok=True)
    code_re = re.compile(code_pattern, re.IGNORECASE)
    die_class_id = _get_die_class_id(model)

    for image_path in image_paths:
        img = cv2.imread(str(image_path))
        if img is None:
            reports.append(
                {
                    "image_path": str(image_path),
                    "status": "failed",
                    "reason": "image_read_failed",
                    "detections": [],
                }
            )
            continue

        results = model(str(image_path), conf=conf_threshold, verbose=False, imgsz=640)
        result = results[0]
        names = result.names if hasattr(result, "names") else {}
        die_candidates = []
        other_candidates = []

        for box in result.boxes:
            class_id = int(box.cls.item()) if box.cls is not None else -1
            class_name = names.get(class_id, str(class_id)) if class_id >= 0 else "unknown"
            conf = float(box.conf.item()) if box.conf is not None else 0.0
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue

            area = (x2 - x1) * (y2 - y1)
            if area < int(min_box_area):
                continue
            item = (area, class_id, class_name, conf, x1, y1, x2, y2)
            if class_name.lower() == "die" and (die_class_id is None or class_id == die_class_id):
                die_candidates.append(item)
            else:
                other_candidates.append(item)

        if max_die_boxes and max_die_boxes > 0 and len(die_candidates) > max_die_boxes:
            die_candidates = sorted(die_candidates, key=lambda x: x[0], reverse=True)[:max_die_boxes]

        candidates = die_candidates + other_candidates

        detections = []
        for _area, class_id, class_name, conf, x1, y1, x2, y2 in candidates:
            crop = img[y1:y2, x1:x2]
            crop = _resize_crop_for_ocr(crop, max_side=640)
            should_ocr = class_name.lower() == "die"
            if ocr_engine is not None and should_ocr:
                text = _extract_text_with_engine(ocr_engine, crop).strip()
                ocr_texts = [text] if text else []
                ocr_codes = [m.group(0).upper() for m in code_re.finditer(text)] if text else []
                if text.isdigit():
                    synthetic_code = f"TH{text.zfill(2)}"
                    if code_re.fullmatch(synthetic_code):
                        ocr_codes.append(synthetic_code)
                ocr_codes = sorted(list({c for c in ocr_codes if c}))
            else:
                ocr_texts = []
                ocr_codes = []

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(conf, 4),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "ocr_texts": ocr_texts,
                    "ocr_codes": ocr_codes,
                }
            )

        annotated = result.plot()
        annotated_path = annotated_dir / f"{image_path.stem}_annotated{image_path.suffix}"
        cv2.imwrite(str(annotated_path), annotated)

        reports.append(
            {
                "image_path": str(image_path),
                "status": "ok",
                "detection_count": len(detections),
                "annotated_image_path": str(annotated_path),
                "detections": detections,
            }
        )

    return reports
