#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR


ImageInput = Union[str, Path, np.ndarray]
SLOT_CODE_ORDER = ("TH26", "TH25", "TH20", "TH16", "TH12")
SLOT_ORDER_INDEX = {code: idx for idx, code in enumerate(SLOT_CODE_ORDER)}


class RapidDigitOCR:
    """RapidOCR based recognizer for die number crops."""

    def __init__(self, text_score: float = 0.8, code_prefix: str = "TH") -> None:
        self.text_score = float(text_score)
        self.code_prefix = str(code_prefix).upper()
        self.ocr = RapidOCR(text_score=self.text_score)

    def warmup(self) -> None:
        dummy = np.zeros((64, 160, 3), dtype=np.uint8)
        self.recognize_die_code(dummy)

    def recognize_die_code(
        self,
        crop: np.ndarray,
        expected_codes: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        texts, numbers, elapsed = self.extract_numbers(crop)
        codes = self._numbers_to_codes(numbers, expected_codes=expected_codes)
        return {
            "texts": texts,
            "codes": codes,
            "numbers": numbers,
            "elapsed": elapsed,
            "code_scores": {},
            "engine": "rapidocr_onnxruntime",
        }

    def extract_numbers(self, image: ImageInput) -> Tuple[List[str], List[str], float]:
        start = time.perf_counter()
        texts: List[str] = []
        numbers: List[str] = []

        for candidate in self._build_variants(image):
            try:
                result, _elapse = self.ocr(candidate)
            except Exception:
                continue
            candidate_texts: List[str] = []
            candidate_numbers: List[str] = []
            for text, _conf in _iter_rapidocr_texts(result):
                if text:
                    candidate_texts.append(text)
                    candidate_numbers.extend(re.findall(r"\d+", text))
            texts.extend(candidate_texts)
            numbers.extend(candidate_numbers)
            has_code_like_text = any(re.search(r"[A-Z]{1,3}\s*\d{2}", t, re.IGNORECASE) for t in candidate_texts)
            has_two_digit_number = any(len(re.sub(r"\D", "", n)) >= 2 for n in candidate_numbers)
            if has_code_like_text or has_two_digit_number:
                break

        elapsed = time.perf_counter() - start
        return _unique_keep_order(texts), _unique_keep_order(numbers), elapsed

    def _build_variants(self, image: ImageInput) -> List[ImageInput]:
        if isinstance(image, (str, Path)):
            return [str(image)]
        if image is None or getattr(image, "size", 0) == 0:
            return []

        base = _scale_keep_aspect(image, target_long_side=900)
        variants: List[np.ndarray] = []
        h, w = base.shape[:2]
        if h >= 16 and w >= 16:
            rois = [
                # The top-right slots (TH25/TH26) often have small, slanted text.
                # Try raw high-resolution lower/top-ring regions before enhanced
                # full crops; downscaling those crops makes RapidOCR miss them.
                base[int(h * 0.50): min(h, int(h * 0.96)), 0:min(w, int(w * 0.82))],
                base[int(h * 0.55): h, 0:min(w, int(w * 0.76))],
                base[0:min(h, int(h * 0.50)), max(0, int(w * 0.35)):w],
                base[int(h * 0.18):min(h, int(h * 0.92)), max(0, int(w * 0.12)):min(w, int(w * 0.90))],
                base,
            ]
        else:
            rois = [base]

        for roi in rois:
            if roi is None or getattr(roi, "size", 0) == 0:
                continue
            variants.append(roi)
            variants.append(cv2.rotate(roi, cv2.ROTATE_180))
            variants.append(cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE))
            variants.append(cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE))
            norm = _enhance_for_ocr(roi)
            variants.append(norm)

        return variants

    def _numbers_to_codes(
        self,
        numbers: Sequence[str],
        expected_codes: Optional[Sequence[str]] = None,
    ) -> List[str]:
        expected = {
            str(code).strip().upper()
            for code in (expected_codes or [])
            if str(code).strip()
        }
        codes: List[str] = []
        for raw in numbers:
            digits = re.sub(r"\D", "", str(raw))
            if not digits:
                continue
            candidates = []
            if len(digits) <= 2:
                candidates.append(digits.zfill(2))
            else:
                for idx in range(0, len(digits) - 1):
                    candidates.append(digits[idx:idx + 2])
            for pair in candidates:
                code = f"{self.code_prefix}{pair}"
                if expected and code not in expected:
                    continue
                codes.append(code)
        return _unique_keep_order(codes)


def extract_numbers_from_image(image_path: ImageInput) -> Tuple[List[str], float]:
    recognizer = RapidDigitOCR(text_score=0.8)
    _texts, numbers, elapsed = recognizer.extract_numbers(image_path)
    return numbers, elapsed


def detect_and_ocr_with_det(
    image_paths: List[Path],
    model,
    ocr_engine,
    conf_threshold: float,
    class_conf_thresholds: Optional[Dict[str, float]],
    code_pattern: str,
    annotated_dir: Path,
    min_box_area: int = 900,
    max_die_boxes: int = 0,
    ocr_timeout_ms: int = 8000,
    max_ocr_items: int = 0,
    expected_die_codes: Optional[List[str]] = None,
    detect_imgsz: int = 768,
) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    annotated_dir.mkdir(parents=True, exist_ok=True)
    code_re = re.compile(code_pattern, re.IGNORECASE)
    expected_set = {str(code).strip().upper() for code in (expected_die_codes or []) if str(code).strip()}

    for image_path in image_paths:
        t0 = cv2.getTickCount()
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

        td0 = cv2.getTickCount()
        results = model(img, conf=float(conf_threshold), verbose=False, imgsz=max(320, int(detect_imgsz)))
        td1 = cv2.getTickCount()
        result = results[0]
        names = result.names if hasattr(result, "names") else {}

        die_candidates = []
        other_candidates = []
        for box in result.boxes:
            class_id = int(box.cls.item()) if box.cls is not None else -1
            class_name = names.get(class_id, str(class_id)) if class_id >= 0 else "unknown"
            conf = float(box.conf.item()) if box.conf is not None else 0.0
            class_conf = float(conf_threshold)
            if isinstance(class_conf_thresholds, dict):
                try:
                    class_conf = float(class_conf_thresholds.get(str(class_name).strip().lower(), conf_threshold))
                except Exception:
                    class_conf = float(conf_threshold)
            if conf < class_conf:
                continue

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
            if str(class_name).lower() == "die":
                die_candidates.append(item)
            else:
                other_candidates.append(item)

        if max_die_boxes and max_die_boxes > 0 and len(die_candidates) > max_die_boxes:
            die_candidates = sorted(die_candidates, key=lambda x: x[0], reverse=True)[:max_die_boxes]
        candidates = die_candidates + other_candidates

        to0 = cv2.getTickCount()
        detections = []
        crop_dir = annotated_dir.parent / "ocr_crops" / image_path.stem
        crop_dir.mkdir(parents=True, exist_ok=True)
        die_ocr_count = 0
        deadline = time.perf_counter() + max(0.5, float(ocr_timeout_ms) / 1000.0)

        for idx, (_area, class_id, class_name, conf, x1, y1, x2, y2) in enumerate(candidates, start=1):
            ocr_texts: List[str] = []
            ocr_codes: List[str] = []
            crop_path = ""
            recognized_code = ""
            expected_slot_code = ""
            slot_index = None
            slot_match = False

            if ocr_engine is not None and str(class_name).lower() == "die":
                if max_ocr_items > 0 and die_ocr_count >= int(max_ocr_items):
                    pass
                elif time.perf_counter() < deadline:
                    px1, py1, px2, py2 = _pad_bbox(x1, y1, x2, y2, img.shape[1], img.shape[0], pad_ratio=0.24)
                    crop = img[py1:py2, px1:px2]
                    crop_file = crop_dir / f"die_{idx:02d}_{x1}_{y1}_{x2}_{y2}_ocr.jpg"
                    cv2.imwrite(str(crop_file), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    crop_path = str(crop_file)
                    die_ocr_count += 1

                    raw = ocr_engine.recognize_die_code(crop, expected_codes=expected_die_codes)
                    ocr_texts = [str(t).strip().upper() for t in (raw.get("texts", []) or []) if str(t).strip()]
                    ocr_codes = [str(c).strip().upper() for c in (raw.get("codes", []) or []) if str(c).strip()]
                    for text in ocr_texts:
                        for match in code_re.finditer(text):
                            ocr_codes.append(match.group(0).upper())
                    if expected_set:
                        ocr_codes = [code for code in ocr_codes if code in expected_set]
                    ocr_codes = _unique_keep_order(ocr_codes)
                    recognized_code = ocr_codes[0] if ocr_codes else ""
                    if recognized_code in SLOT_ORDER_INDEX:
                        expected_slot_code = recognized_code
                        slot_index = SLOT_ORDER_INDEX[recognized_code]
                        slot_match = True

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(conf, 4),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "ocr_texts": ocr_texts,
                    "ocr_codes": ocr_codes,
                    "expected_slot_code": expected_slot_code,
                    "recognized_code": recognized_code,
                    "slot_index": slot_index,
                    "slot_match": slot_match,
                    "ocr_crop_path": crop_path,
                }
            )

        to1 = cv2.getTickCount()
        detections = sorted(detections, key=_detection_slot_order)

        ta0 = cv2.getTickCount()
        annotated = result.plot()
        annotated_path = annotated_dir / f"{image_path.stem}_annotated{image_path.suffix}"
        cv2.imwrite(str(annotated_path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        ta1 = cv2.getTickCount()

        t1 = cv2.getTickCount()
        hz = cv2.getTickFrequency()
        reports.append(
            {
                "image_path": str(image_path),
                "status": "ok",
                "detection_count": len(detections),
                "annotated_image_path": str(annotated_path),
                "timings_ms": {
                    "detect": round((td1 - td0) * 1000.0 / hz, 2),
                    "ocr": round((to1 - to0) * 1000.0 / hz, 2),
                    "annotate": round((ta1 - ta0) * 1000.0 / hz, 2),
                    "total": round((t1 - t0) * 1000.0 / hz, 2),
                },
                "detections": detections,
            }
        )
    return reports


def _iter_rapidocr_texts(result: Any) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    if not result:
        return out
    for line in result:
        try:
            text = str(line[1]).strip()
            conf = float(line[2])
        except Exception:
            continue
        if text:
            out.append((text, conf))
    return out


def _pad_bbox(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    img_w: int,
    img_h: int,
    pad_ratio: float = 0.24,
) -> Tuple[int, int, int, int]:
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    px = int(w * pad_ratio)
    py = int(h * pad_ratio)
    return (
        max(0, x1 - px),
        max(0, y1 - py),
        min(img_w, x2 + px),
        min(img_h, y2 + py),
    )


def _detection_slot_order(det: Dict[str, Any]) -> Tuple[int, int]:
    if str(det.get("class_name", "")).lower() != "die":
        return (len(SLOT_CODE_ORDER) + 1, 0)
    slot_index = det.get("slot_index")
    if isinstance(slot_index, int):
        return (slot_index, 0)
    for code in det.get("ocr_codes", []) or []:
        code = str(code).strip().upper()
        if code in SLOT_ORDER_INDEX:
            return (SLOT_ORDER_INDEX[code], 0)
    return (len(SLOT_CODE_ORDER), 0)


def _resize_keep_aspect(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    return cv2.resize(
        img,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_CUBIC,
    )


def _scale_keep_aspect(img: np.ndarray, target_long_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= 0 or longest == target_long_side:
        return img
    scale = target_long_side / float(longest)
    return cv2.resize(
        img,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA,
    )


def _enhance_for_ocr(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (3, 3), 0)
    sharp = cv2.addWeighted(clahe, 1.6, blur, -0.6, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


def _unique_keep_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        value = str(item).strip().upper()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract numbers from images with RapidOCR.")
    parser.add_argument("images", nargs="+", help="Image paths to recognize.")
    args = parser.parse_args()

    recognizer = RapidDigitOCR(text_score=0.8)
    for path in args.images:
        texts, numbers, elapsed = recognizer.extract_numbers(path)
        print(f"{path}: numbers={numbers} texts={texts} time={elapsed:.4f}s")
