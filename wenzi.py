import os
import glob
import re
import time
from itertools import combinations, permutations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from ultralytics import YOLO

os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_use_mkldnn", "0")

from paddleocr import PaddleOCR


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "best_last.pt"
DEFAULT_IMAGE_FOLDER = PROJECT_ROOT
CONF_THRESH = 0.45
OUTPUT_LOG = PROJECT_ROOT / "inference_log.txt"
CAPTURE_PATTERN = "capture_clear_*.jpg"
SLOT_CODE_ORDER = ("TH26", "TH25", "TH20", "TH16", "TH12")
SLOT_ORDER_INDEX = {code: idx for idx, code in enumerate(SLOT_CODE_ORDER)}


def _resize_keep_aspect(img, max_side: int = 960):
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    ratio = max_side / float(longest)
    new_w = max(1, int(w * ratio))
    new_h = max(1, int(h * ratio))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _pad_bbox(
    x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int, pad_ratio: float = 0.24
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


def _build_ocr_variants(crop, heavy: bool = False) -> List:
    if crop is None or getattr(crop, "size", 0) == 0:
        return []

    base = _resize_keep_aspect(crop, max_side=1280)
    # Upscale tiny crops to make engraved characters more readable.
    h0, w0 = base.shape[:2]
    min_side = min(h0, w0)
    if min_side < 140:
        scale = min(3.0, 140.0 / float(max(1, min_side)))
        nw = max(1, int(w0 * scale))
        nh = max(1, int(h0 * scale))
        base = cv2.resize(base, (nw, nh), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY) if len(base.shape) == 3 else base
    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (3, 3), 0)
    # Unsharp mask can improve thin stroke contrast on metal parts.
    sharp = cv2.addWeighted(clahe, 1.5, blur, -0.5, 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = cv2.bitwise_not(otsu)

    base_list = [
        base,
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR),
    ]

    out = []
    if not heavy:
        # Fast pass: keep only the most useful low-cost variants.
        out.extend([base_list[0], base_list[2]])
        return out

    # Heavy pass for hard cases only.
    out.extend(
        [
            base_list[0],
            base_list[2],
            base_list[3],
            base_list[4],
            base_list[5],
            cv2.rotate(base_list[0], cv2.ROTATE_180),
            cv2.rotate(base_list[2], cv2.ROTATE_180),
        ]
    )
    return out


def _texts_from_predict_output(out) -> List[str]:
    texts = []
    if isinstance(out, dict):
        texts.extend([str(t).strip() for t in (out.get("rec_texts") or []) if str(t).strip()])
    elif hasattr(out, "rec_texts") and getattr(out, "rec_texts"):
        texts.extend([str(t).strip() for t in out.rec_texts if str(t).strip()])
    elif isinstance(out, list):
        for line in out:
            try:
                txt = str(line[1][0]).strip()
                if txt:
                    texts.append(txt)
            except Exception:
                continue
    return texts


def _texts_from_ocr_raw(raw) -> List[str]:
    texts = []
    if not isinstance(raw, list):
        return texts
    for block in raw:
        if isinstance(block, dict):
            rec = block.get("rec_texts") or []
            texts.extend([str(t).strip() for t in rec if str(t).strip()])
            continue
        if not isinstance(block, list):
            continue
        for line in block:
            try:
                txt = str(line[1][0]).strip()
                if txt:
                    texts.append(txt)
            except Exception:
                continue
    return texts


def _extract_texts_with_engine(
    engine,
    crop,
    code_re,
    heavy: bool = False,
    budget_s: float = 1.0,
    expected_die_codes: Optional[List[str]] = None,
) -> List[str]:
    texts = []
    seen = set()
    t_start = time.perf_counter()
    did_det_rec = False
    for var in _build_ocr_variants(crop, heavy=heavy):
        if (time.perf_counter() - t_start) >= max(0.1, float(budget_s)):
            break
        try:
            outs = list(engine.predict(var))
        except Exception:
            outs = []
        for out in outs:
            for t in _texts_from_predict_output(out):
                if t not in seen:
                    seen.add(t)
                    texts.append(t)
            if _extract_codes_from_texts_with_expected(
                texts, code_re, expected_die_codes=expected_die_codes
            ):
                return texts

        # det+rec is expensive: run at most once in heavy pass.
        if heavy and (not did_det_rec):
            did_det_rec = True
            try:
                raw = engine.ocr(var, det=True, rec=True, cls=False)
            except Exception:
                raw = []
            for t in _texts_from_ocr_raw(raw):
                if t not in seen:
                    seen.add(t)
                    texts.append(t)
            if _extract_codes_from_texts_with_expected(
                texts, code_re, expected_die_codes=expected_die_codes
            ):
                return texts
    return texts


def _build_die_subcrops(crop) -> List:
    if crop is None or getattr(crop, "size", 0) == 0:
        return []
    h, w = crop.shape[:2]
    if h < 8 or w < 8:
        return [crop]
    # Prioritize regions where die code is usually engraved.
    bottom = crop[min(h - 4, int(h * 0.54)) : h, :]
    lower_mid = crop[min(h - 4, int(h * 0.42)) : min(h, int(h * 0.88)), max(0, int(w * 0.12)) : min(w, int(w * 0.88))]
    center = crop[max(0, int(h * 0.2)) : min(h, int(h * 0.84)), max(0, int(w * 0.15)) : min(w, int(w * 0.85))]
    out = [crop, bottom, lower_mid, center]
    return [x for x in out if x is not None and getattr(x, "size", 0) > 0]


def _extract_codes_from_texts(texts: List[str], code_re) -> List[str]:
    return _extract_codes_from_texts_with_expected(texts, code_re, expected_die_codes=None)


def _detection_slot_order(det: Dict) -> Tuple[int, int]:
    if str(det.get("class_name", "")).lower() != "die":
        return (len(SLOT_CODE_ORDER) + 1, 0)
    slot_index = det.get("slot_index")
    if isinstance(slot_index, int):
        return (slot_index, 0)
    codes = [str(code).strip().upper() for code in det.get("ocr_codes", []) if str(code).strip()]
    for code in codes:
        if code in SLOT_ORDER_INDEX:
            return (SLOT_ORDER_INDEX[code], 0)
    return (len(SLOT_CODE_ORDER), 0)


def _extract_codes_from_texts_with_expected(
    texts: List[str],
    code_re,
    expected_die_codes: Optional[List[str]] = None,
) -> List[str]:
    expected_set = {str(x).strip().upper() for x in (expected_die_codes or []) if str(x).strip()}

    # Common OCR confusions for stamped metal text.
    char_to_digit = {
        "0": ["0"],
        "O": ["0"],
        "Q": ["0"],
        "D": ["0"],
        "1": ["1"],
        "I": ["1"],
        "L": ["1"],
        "T": ["1"],
        "2": ["2"],
        "Z": ["2"],
        "H": ["2"],
        "5": ["5"],
        "S": ["5"],
        "6": ["6"],
        "G": ["6"],
        "8": ["8"],
        "B": ["8"],
    }
    digit_accept = {
        "0": set("0OQD"),
        "1": set("1ILT"),
        "2": set("2ZH"),
        "5": set("5S"),
        "6": set("6G"),
        "8": set("8B"),
    }

    def _norm_token(s: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", str(s).upper())

    def _expand_suffix_to_codes(sfx: str) -> List[str]:
        sfx = _norm_token(sfx)
        if not sfx:
            return []
        # pick first two chars as candidate suffix
        if len(sfx) == 1:
            chars = [sfx[0], sfx[0]]
        else:
            chars = [sfx[0], sfx[1]]
        d1_opts = char_to_digit.get(chars[0], [chars[0]] if chars[0].isdigit() else [])
        d2_opts = char_to_digit.get(chars[1], [chars[1]] if chars[1].isdigit() else [])
        out = []
        for a in d1_opts:
            for b in d2_opts:
                if a.isdigit() and b.isdigit():
                    out.append(f"TH{a}{b}")
        return out

    codes = []
    codes = []
    for text in texts:
        s = str(text)
        for m in code_re.finditer(s):
            codes.append(m.group(0).upper())
        # digits-only outputs from OCR are still real OCR output.
        for grp in re.findall(r"\d+", s):
            if not grp:
                continue
            if 1 <= len(grp) <= 2:
                c = f"TH{grp.zfill(2)}"
                if code_re.fullmatch(c):
                    codes.append(c)
            elif len(grp) % 2 == 0 and len(grp) <= 8:
                for i in range(0, len(grp), 2):
                    c = f"TH{grp[i:i+2]}"
                    if code_re.fullmatch(c):
                        codes.append(c)

        # Parse TH-like noisy tokens (e.g., THHO -> TH20).
        for token in re.findall(r"TH[A-Z0-9]{1,4}", _norm_token(s)):
            suffix = token[2:]
            codes.extend(_expand_suffix_to_codes(suffix))

        # If expected set is provided, fuzzy-match noisy THxx tokens to expected codes.
        if expected_set:
            for token in re.findall(r"TH[A-Z0-9]{2}", _norm_token(s)):
                sfx = token[2:4]
                if len(sfx) != 2:
                    continue
                for exp in expected_set:
                    d1, d2 = exp[2], exp[3]
                    c1, c2 = sfx[0], sfx[1]
                    ok1 = c1 == d1 or c1 in digit_accept.get(d1, set())
                    ok2 = c2 == d2 or c2 in digit_accept.get(d2, set())
                    if ok1 and ok2:
                        codes.append(exp)
    uniq = sorted(list({c for c in codes if c}))
    if expected_set:
        uniq = [c for c in uniq if c in expected_set]
    return uniq


def detect_and_ocr_with_wenzi(
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
    max_ocr_items: int = 5,
    expected_die_codes: Optional[List[str]] = None,
    detect_imgsz: int = 768,
) -> List[Dict]:
    def _assign_expected_slots(
        fast_die_results: List[Dict],
        expected_codes: Optional[List[str]],
        position_priors: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        expected = [str(code).strip().upper() for code in (expected_codes or []) if str(code).strip()]
        if not fast_die_results or not expected:
            return

        priors = {
            (code if code.upper().startswith("TH") else f"TH{code}").upper(): pos
            for code, pos in (position_priors or {}).items()
            if isinstance(pos, tuple) and len(pos) == 2
        }
        if not priors:
            return

        best_total = None
        best_pairs = None
        item_indexes = list(range(len(fast_die_results)))
        prior_codes = [code for code in SLOT_CODE_ORDER if code in priors]
        assign_count = min(len(item_indexes), len(prior_codes))
        for chosen_items in combinations(item_indexes, assign_count):
            for code_perm in permutations(prior_codes, assign_count):
                total = 0.0
                pairs = []
                for idx, code in zip(chosen_items, code_perm):
                    cx, cy = fast_die_results[idx].get("center_norm", (None, None))
                    if cx is None or cy is None:
                        total += 999.0
                        continue
                    px, py = priors[code]
                    total += (float(cx) - float(px)) ** 2 + (float(cy) - float(py)) ** 2
                    pairs.append((idx, code))
                if best_total is None or total < best_total:
                    best_total = total
                    best_pairs = pairs
        if best_pairs:
            for idx, code in best_pairs:
                actual_codes = [
                    str(c).strip().upper()
                    for c in fast_die_results[idx].get("ocr_codes", [])
                    if str(c).strip()
                ]
                fast_die_results[idx]["expected_slot_code"] = code
                fast_die_results[idx]["slot_index"] = SLOT_ORDER_INDEX.get(code)
                fast_die_results[idx]["recognized_code"] = actual_codes[0] if actual_codes else ""
                fast_die_results[idx]["slot_match"] = bool(actual_codes and actual_codes[0] == code)

    reports: List[Dict] = []
    annotated_dir.mkdir(parents=True, exist_ok=True)
    code_re = re.compile(code_pattern, re.IGNORECASE)

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
        imgsz = max(320, int(detect_imgsz))
        results = model(img, conf=float(conf_threshold), verbose=False, imgsz=imgsz)
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
            if class_name.lower() == "die":
                die_candidates.append(item)
            else:
                other_candidates.append(item)

        if max_die_boxes and max_die_boxes > 0 and len(die_candidates) > max_die_boxes:
            die_candidates = sorted(die_candidates, key=lambda x: x[0], reverse=True)[:max_die_boxes]
        candidates = die_candidates + other_candidates

        to0 = cv2.getTickCount()
        fast_die_results: Dict[Tuple[int, int, int, int], Dict] = {}
        if ocr_engine is not None and hasattr(ocr_engine, "recognize_die_code"):
            staged = []
            for _area, class_id, class_name, conf, x1, y1, x2, y2 in candidates:
                if class_name.lower() != "die":
                    continue
                px1, py1, px2, py2 = _pad_bbox(x1, y1, x2, y2, img.shape[1], img.shape[0], pad_ratio=0.24)
                crop = img[py1:py2, px1:px2]
                fast_result = ocr_engine.recognize_die_code(crop, expected_codes=expected_die_codes)
                staged.append(
                    {
                        "key": (x1, y1, x2, y2),
                        "ocr_texts": list(fast_result.get("texts", []) or []),
                        "ocr_codes": list(fast_result.get("codes", []) or []),
                        "code_scores": dict(fast_result.get("code_scores", {}) or {}),
                        "center_norm": (
                            ((x1 + x2) * 0.5) / max(1.0, float(img.shape[1])),
                            ((y1 + y2) * 0.5) / max(1.0, float(img.shape[0])),
                        ),
                    }
                )
            _assign_expected_slots(
                staged,
                expected_die_codes,
                position_priors=getattr(ocr_engine, "position_priors", None),
            )
            fast_die_results = {item["key"]: item for item in staged}

        detections = []
        img_h, img_w = img.shape[:2]
        ocr_budget_total_s = max(0.5, float(ocr_timeout_ms) / 1000.0)
        ocr_budget_start = time.perf_counter()
        die_ocr_count = 0
        die_total = sum(1 for _, _, cname, _, _, _, _, _ in candidates if str(cname).lower() == "die")
        per_die_budget_s = 0.0
        if die_total > 0:
            # Fair-share OCR budget: avoid first few die boxes consuming all time.
            per_die_budget_s = ocr_budget_total_s / float(die_total)
            per_die_budget_s = max(0.65, min(1.6, per_die_budget_s * 1.05))

        for _area, class_id, class_name, conf, x1, y1, x2, y2 in candidates:
            ocr_texts = []
            ocr_codes = []
            if ocr_engine is not None and class_name.lower() == "die":
                if hasattr(ocr_engine, "recognize_die_code"):
                    fast_result = fast_die_results.get((x1, y1, x2, y2), {})
                    ocr_texts = list(fast_result.get("ocr_texts", []) or [])
                    ocr_codes = list(fast_result.get("ocr_codes", []) or [])
                    expected_slot_code = str(fast_result.get("expected_slot_code", "") or "")
                    recognized_code = str(fast_result.get("recognized_code", "") or "")
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
                            "slot_index": fast_result.get("slot_index"),
                            "slot_match": bool(fast_result.get("slot_match", False)),
                        }
                    )
                    continue
                if max_ocr_items > 0 and die_ocr_count >= int(max_ocr_items):
                    detections.append(
                        {
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": round(conf, 4),
                            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                            "ocr_texts": [],
                            "ocr_codes": [],
                        }
                    )
                    continue
                elapsed = time.perf_counter() - ocr_budget_start
                remain_s = ocr_budget_total_s - elapsed
                if remain_s <= 0.0:
                    detections.append(
                        {
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": round(conf, 4),
                            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                            "ocr_texts": [],
                            "ocr_codes": [],
                        }
                    )
                    continue
                die_start = time.perf_counter()
                die_budget_s = min(remain_s, per_die_budget_s) if per_die_budget_s > 0 else remain_s
                die_budget_s = max(0.45, die_budget_s)
                px1, py1, px2, py2 = _pad_bbox(x1, y1, x2, y2, img_w, img_h, pad_ratio=0.24)
                crop = img[py1:py2, px1:px2]
                fast_budget = min(0.55, max(0.14, die_budget_s * 0.45))
                ocr_texts = []
                for sub_crop in _build_die_subcrops(crop):
                    global_elapsed = time.perf_counter() - ocr_budget_start
                    global_remain_s = ocr_budget_total_s - global_elapsed
                    local_elapsed = time.perf_counter() - die_start
                    local_remain_s = die_budget_s - local_elapsed
                    if global_remain_s <= 0.0 or local_remain_s <= 0.0:
                        break
                    sub_fast_budget = min(fast_budget, max(0.08, local_remain_s * 0.45))
                    sub_texts = _extract_texts_with_engine(
                        ocr_engine,
                        sub_crop,
                        code_re,
                        heavy=False,
                        budget_s=sub_fast_budget,
                        expected_die_codes=expected_die_codes,
                    )
                    if sub_texts:
                        seen = set(ocr_texts)
                        for t in sub_texts:
                            if t and t not in seen:
                                seen.add(t)
                                ocr_texts.append(t)
                    sub_codes = _extract_codes_from_texts_with_expected(
                        ocr_texts, code_re, expected_die_codes=expected_die_codes
                    )
                    if sub_codes:
                        break
                ocr_codes = _extract_codes_from_texts_with_expected(
                    ocr_texts, code_re, expected_die_codes=expected_die_codes
                )
                # Only fallback to heavy OCR when fast pass has no code.
                if not ocr_codes:
                    global_elapsed = time.perf_counter() - ocr_budget_start
                    global_remain_s = ocr_budget_total_s - global_elapsed
                    local_elapsed = time.perf_counter() - die_start
                    local_remain_s = die_budget_s - local_elapsed
                    if global_remain_s > 0.12 and local_remain_s > 0.12:
                        heavy_budget = min(0.95, max(0.14, local_remain_s * 0.75))
                        heavy_texts = []
                        for sub_crop in _build_die_subcrops(crop):
                            global_elapsed = time.perf_counter() - ocr_budget_start
                            global_remain_s = ocr_budget_total_s - global_elapsed
                            local_elapsed = time.perf_counter() - die_start
                            local_remain_s = die_budget_s - local_elapsed
                            if global_remain_s <= 0.0 or local_remain_s <= 0.0:
                                break
                            sub_heavy_budget = min(heavy_budget, max(0.1, local_remain_s * 0.55))
                            cur = _extract_texts_with_engine(
                                ocr_engine,
                                sub_crop,
                                code_re,
                                heavy=True,
                                budget_s=sub_heavy_budget,
                                expected_die_codes=expected_die_codes,
                            )
                            if cur:
                                seen = set(heavy_texts)
                                for t in cur:
                                    if t and t not in seen:
                                        seen.add(t)
                                        heavy_texts.append(t)
                            cur_codes = _extract_codes_from_texts_with_expected(
                                heavy_texts, code_re, expected_die_codes=expected_die_codes
                            )
                            if cur_codes:
                                break
                    else:
                        heavy_texts = []
                    if heavy_texts:
                        merged = []
                        seen = set()
                        for t in ocr_texts + heavy_texts:
                            if t and t not in seen:
                                seen.add(t)
                                merged.append(t)
                        ocr_texts = merged
                    ocr_codes = _extract_codes_from_texts_with_expected(
                        ocr_texts, code_re, expected_die_codes=expected_die_codes
                    )
                die_ocr_count += 1

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


def process_images_from_camera(
    image_folder: Optional[Path] = None,
    pattern: str = CAPTURE_PATTERN,
    model_path: Optional[Path] = None,
):
    image_folder = Path(image_folder) if image_folder is not None else DEFAULT_IMAGE_FOLDER
    model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    ocr_engine = PaddleOCR(use_textline_orientation=True, lang="en", enable_mkldnn=False)
    model = YOLO(str(model_path))
    code_re = re.compile(r"TH\d{2}", re.IGNORECASE)

    with OUTPUT_LOG.open("a", encoding="utf-8") as log_file:
        def log_line(msg=""):
            print(msg)
            log_file.write(f"{msg}\n")

        log_line("===== 配件文字识别（海康相机输出）=====")
        log_line()

        files = sorted(glob.glob(str(image_folder / pattern)), key=os.path.getctime, reverse=True)
        if not files:
            log_line(f"[WARN] 未找到匹配模式 '{pattern}' 的图片文件在 {image_folder}")
            return

        for fp in files:
            img = cv2.imread(fp)
            if img is None:
                continue
            log_line(f"========== {Path(fp).name} ==========")
            rep = detect_and_ocr_with_wenzi(
                image_paths=[Path(fp)],
                model=model,
                ocr_engine=ocr_engine,
                conf_threshold=CONF_THRESH,
                code_pattern=r"TH\d{2}",
                annotated_dir=PROJECT_ROOT / "runtime" / "annotated",
                min_box_area=350,
                max_die_boxes=8,
                ocr_timeout_ms=8000,
                max_ocr_items=5,
                expected_die_codes=None,
                detect_imgsz=960,
            )[0]
            for idx, d in enumerate(rep.get("detections", []), 1):
                log_line(
                    f"[{d.get('class_name')} {idx}] OCR文本: "
                    f"{' | '.join(d.get('ocr_texts', [])) if d.get('ocr_texts') else '-'}"
                )
                log_line(
                    f"[{d.get('class_name')} {idx}] OCR编号: "
                    f"{', '.join(d.get('ocr_codes', [])) if d.get('ocr_codes') else '-'}"
                )
            log_line()


if __name__ == "__main__":
    process_images_from_camera()
