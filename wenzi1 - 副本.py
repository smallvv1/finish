import os
import sys
import cv2
import glob
import re
import concurrent.futures
from pathlib import Path
from typing import Dict, List
from ultralytics import YOLO

os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_use_mkldnn", "0")

from paddleocr import PaddleOCR

MODEL_PATH   = r"D:\1\EzYOLO-main\runs\detect\runs\train\exp_2\weights\best.pt" # 浣跨敤鍘熷瀛楃涓查伩鍏嶈浆涔?
#IMAGE_FOLDER = r"D:\1000\pliers_images" # 浣跨敤鍘熷瀛楃涓?
IMAGE_FOLDER = r"D:\1\EzYOLO-main"  # 榛樿涓哄綋鍓嶇洰褰曪紝鎵弿capture_clear_*.jpg

CONF_THRESH  = 0.4
OUTPUT_LOG   = os.path.join(os.path.dirname(__file__), "inference_log.txt")
CAPTURE_PATTERN = "capture_clear_*.jpg"  # 鍖归厤test_hk_opecv.py鐨勮緭鍑烘枃浠?
# ====================================================

# 鍒濆鍖?OCR
ocr = None
model = None


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


def _prepare_ocr_candidates(crop):
    """Generate contrast-enhanced variants for low-contrast engraved text."""
    if crop is None or getattr(crop, "size", 0) == 0:
        return []
    base = crop
    if len(base.shape) == 3:
        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    else:
        gray = base

    # Upscale first: tiny engravings are otherwise unreadable.
    h, w = gray.shape[:2]
    scale = 2.0 if max(h, w) < 220 else 1.5
    up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(up)
    blur = cv2.GaussianBlur(clahe, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = cv2.bitwise_not(otsu)

    # Return BGR images to keep Paddle input shape stable.
    variants = []
    for img in [up, clahe, otsu, inv]:
        variants.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    return variants


def _parse_texts_from_ocr_raw(raw) -> List[str]:
    texts: List[str] = []
    if not isinstance(raw, list):
        return texts
    for item in raw:
        if isinstance(item, list):
            for sub in item:
                if isinstance(sub, (list, tuple)) and len(sub) >= 1:
                    if isinstance(sub[0], str):
                        texts.append(sub[0])
                    elif len(sub) > 1 and isinstance(sub[1], str):
                        texts.append(sub[1])
                elif isinstance(sub, str):
                    texts.append(sub)
        elif isinstance(item, str):
            texts.append(item)
    return [str(t).strip() for t in texts if str(t).strip()]


def _extract_text_items_with_engine(engine, image, min_score: float = 0.3) -> List[Dict]:
    """Run det+rec OCR and return text items with bbox + center."""
    items: List[Dict] = []
    min_score = float(min_score)

    def _append_item(text: str, score: float, xs: List[float], ys: List[float]):
        text = str(text).strip()
        if not text:
            return
        try:
            score_f = float(score)
        except Exception:
            score_f = 0.0
        if score_f < min_score:
            return
        if not xs or not ys:
            return
        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        items.append(
            {
                "text": text,
                "score": score_f,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "center": (cx, cy),
            }
        )

    def _append_from_dict_result(obj: Dict):
        rec_texts = obj.get("rec_texts") or []
        rec_scores = obj.get("rec_scores") or []
        dt_polys = obj.get("dt_polys") or obj.get("rec_polys") or []
        n = min(len(rec_texts), len(dt_polys))
        for i in range(n):
            text = rec_texts[i]
            score = rec_scores[i] if i < len(rec_scores) else 1.0
            poly = dt_polys[i]
            xs, ys = [], []
            if isinstance(poly, (list, tuple)):
                for p in poly:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        try:
                            xs.append(float(p[0]))
                            ys.append(float(p[1]))
                        except Exception:
                            continue
            _append_item(text, score, xs, ys)

    try:
        raw = engine.ocr(image, det=True, rec=True, cls=False)
    except Exception:
        raw = None

    # New PaddleOCR result format (dict list)
    if isinstance(raw, list):
        for block in raw:
            if isinstance(block, dict):
                _append_from_dict_result(block)
                continue
            if not isinstance(block, list):
                continue
            for line in block:
                # Legacy format: [quad_points, (text, score)]
                if not isinstance(line, (list, tuple)) or len(line) < 2:
                    continue
                quad = line[0]
                rec = line[1]
                if not isinstance(rec, (list, tuple)) or len(rec) < 1:
                    continue
                text = rec[0]
                score = rec[1] if len(rec) > 1 else 1.0
                xs, ys = [], []
                if isinstance(quad, (list, tuple)):
                    for p in quad:
                        if isinstance(p, (list, tuple)) and len(p) >= 2:
                            try:
                                xs.append(float(p[0]))
                                ys.append(float(p[1]))
                            except Exception:
                                continue
                _append_item(text, score, xs, ys)

    if items:
        return items

    # Fallback to predict() for versions where ocr() output differs.
    try:
        outputs = list(engine.predict(image))
    except Exception:
        outputs = []
    for out in outputs:
        if isinstance(out, dict):
            _append_from_dict_result(out)
    return items


def _parse_text_from_predict_result(first) -> str:
    if isinstance(first, dict) and first.get("rec_texts"):
        return " ".join(first["rec_texts"]).strip()
    if hasattr(first, "rec_texts") and getattr(first, "rec_texts"):
        return " ".join(first.rec_texts).strip()
    if isinstance(first, list):
        return " ".join([line[1][0] for line in first if len(line) > 1 and len(line[1]) > 0]).strip()
    return ""


def _extract_text_with_engine(engine, crop) -> str:
    for img in _prepare_ocr_candidates(crop):
        # 1) recognition-only path (fast for boxed crops)
        try:
            raw = engine.ocr(img, det=False, rec=True, cls=False)
            text = " ".join(_parse_texts_from_ocr_raw(raw)).strip()
            if text:
                return text
        except Exception:
            pass

        # 2) generic predict path
        try:
            res = list(engine.predict(img))
            if res:
                text = _parse_text_from_predict_result(res[0])
                if text:
                    return text
        except Exception:
            pass

        # 3) full detect+recognize fallback for extreme low-contrast text
        try:
            raw = engine.ocr(img, det=True, rec=True, cls=False)
            text = " ".join(_parse_texts_from_ocr_raw(raw)).strip()
            if text:
                return text
        except Exception:
            pass
    return ""


def _expand_bbox(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int, ratio: float = 0.18):
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    px = int(w * ratio)
    py = int(h * ratio)
    return (
        max(0, x1 - px),
        max(0, y1 - py),
        min(img_w, x2 + px),
        min(img_h, y2 + py),
    )


def _extract_texts_with_engine(engine, crops) -> List[str]:
    if not crops:
        return []
    # Prefer batched inference first to reduce per-call overhead.
    try:
        outputs = list(engine.predict(crops))
    except Exception:
        return [_extract_text_with_engine(engine, crop) for crop in crops]

    texts: List[str] = []
    for first in outputs:
        texts.append(_parse_text_from_predict_result(first))
    if len(texts) == len(crops):
        return texts

    # Fallback: robust per-crop extraction if batch output is incomplete.
    try:
        return [_extract_text_with_engine(engine, crop) for crop in crops]
    except Exception:
        return texts


def _extract_codes_from_text(text: str, code_re) -> List[str]:
    if not text:
        return []
    codes = [m.group(0).upper() for m in code_re.finditer(text)]

    # Backward compatibility: OCR may output only digits, e.g. "20" or "2020".
    # Convert digit groups to TH codes when they satisfy the configured pattern.
    digit_groups = re.findall(r"\d+", text)
    for grp in digit_groups:
        if not grp:
            continue
        if 1 <= len(grp) <= 2:
            candidate = f"TH{grp.zfill(2)}"
            if code_re.fullmatch(candidate):
                codes.append(candidate)
            continue
        if len(grp) % 2 == 0 and len(grp) <= 8:
            for i in range(0, len(grp), 2):
                part = grp[i : i + 2]
                candidate = f"TH{part}"
                if code_re.fullmatch(candidate):
                    codes.append(candidate)

    return sorted(list({c for c in codes if c}))


def _run_with_timeout(func, timeout_sec: float, *args):
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(func, *args)
    try:
        return fut.result(timeout=max(0.2, float(timeout_sec)))
    finally:
        # Do not wait for unfinished OCR task to complete.
        ex.shutdown(wait=False, cancel_futures=True)


def detect_and_ocr_with_wenzi(
    image_paths: List[Path],
    model,
    ocr_engine,
    conf_threshold: float,
    class_conf_thresholds: Dict[str, float],
    code_pattern: str,
    annotated_dir: Path,
    min_box_area: int = 900,
    max_die_boxes: int = 0,
    imgsz: int = 640,
    ocr_fallback_full: bool = False,
    ocr_fast_max_side: int = 256,
    ocr_timeout_ms: int = 8000,
    max_ocr_items: int = 0,
) -> List[Dict]:
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
        # Reuse the already-decoded frame to avoid duplicate disk decode.
        results = model(img, conf=conf_threshold, verbose=False, imgsz=int(imgsz))
        td1 = cv2.getTickCount()
        result = results[0]
        names = result.names if hasattr(result, "names") else {}
        die_candidates = []
        other_candidates = []

        for box in result.boxes:
            class_id = int(box.cls.item()) if box.cls is not None else -1
            class_name = names.get(class_id, str(class_id)) if class_id >= 0 else "unknown"
            conf = float(box.conf.item()) if box.conf is not None else 0.0
            class_key = str(class_name).strip().lower()
            class_thresh = float(class_conf_thresholds.get(class_key, conf_threshold))
            if conf < class_thresh:
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
            # Prefer higher confidence boxes first, area as tie-breaker.
            die_candidates = sorted(die_candidates, key=lambda x: (x[3], x[0]), reverse=True)[:max_die_boxes]
        candidates = die_candidates + other_candidates

        die_crops_fast = []
        die_crops_full = []
        die_indexes = []
        die_items = []
        for i, (area, _cid, cname, conf, x1, y1, x2, y2) in enumerate(candidates):
            if cname.lower() != "die":
                continue
            die_items.append((i, float(conf), int(area), x1, y1, x2, y2))

        if max_ocr_items and max_ocr_items > 0 and len(die_items) > max_ocr_items:
            die_items = sorted(die_items, key=lambda x: (x[1], x[2]), reverse=True)[:max_ocr_items]

        for i, _conf, _area, x1, y1, x2, y2 in die_items:
            crop = img[y1:y2, x1:x2]
            die_crops_fast.append(_resize_crop_for_ocr(crop, max_side=max(96, int(ocr_fast_max_side))))
            if ocr_fallback_full:
                die_crops_full.append(_resize_crop_for_ocr(crop, max_side=640))
            die_indexes.append(i)

        die_texts: Dict[int, str] = {}
        die_codes: Dict[int, List[str]] = {}
        to0 = cv2.getTickCount()
        if ocr_engine is not None and die_crops_fast:
            timeout_sec = max(0.2, float(ocr_timeout_ms) / 1000.0)
            timed_out = False
            try:
                fast_texts = _run_with_timeout(_extract_texts_with_engine, timeout_sec, ocr_engine, die_crops_fast)
            except concurrent.futures.TimeoutError:
                timed_out = True
                fast_texts = []
            except Exception:
                fast_texts = []
            fallback_pos = []
            for pos, (idx, txt) in enumerate(zip(die_indexes, fast_texts)):
                text = str(txt).strip() if txt is not None else ""
                # Always keep OCR text for UI visibility, even when no code is parsed.
                if text:
                    die_texts[idx] = text
                    parsed_codes = _extract_codes_from_text(text, code_re)
                    if parsed_codes:
                        die_codes[idx] = parsed_codes
                if not _extract_codes_from_text(text, code_re):
                    fallback_pos.append(pos)
            if (not timed_out) and ocr_fallback_full and fallback_pos:
                fallback_crops = [die_crops_full[pos] for pos in fallback_pos]
                try:
                    fallback_texts = _run_with_timeout(
                        _extract_texts_with_engine, timeout_sec, ocr_engine, fallback_crops
                    )
                except Exception:
                    fallback_texts = []
                for pos, txt in zip(fallback_pos, fallback_texts):
                    text = str(txt).strip() if txt is not None else ""
                    if text:
                        die_texts[die_indexes[pos]] = text
                        parsed_codes = _extract_codes_from_text(text, code_re)
                        if parsed_codes:
                            die_codes[die_indexes[pos]] = parsed_codes

        # If per-box OCR misses codes, run one full-image OCR pass and map true OCR text
        # to each die box by spatial overlap. This is still pure OCR output, no guessing.
        if ocr_engine is not None and candidates:
            missing_die_idx = []
            for i, (_area, _cid, cname, _conf, _x1, _y1, _x2, _y2) in enumerate(candidates):
                if cname.lower() != "die":
                    continue
                if die_codes.get(i):
                    continue
                missing_die_idx.append(i)

            if missing_die_idx:
                img_h, img_w = img.shape[:2]
                global_items = _extract_text_items_with_engine(ocr_engine, img, min_score=0.25)
                for i in missing_die_idx:
                    _area, _cid, _cname, _conf, x1, y1, x2, y2 = candidates[i]
                    ex1, ey1, ex2, ey2 = _expand_bbox(x1, y1, x2, y2, img_w, img_h, ratio=0.22)
                    texts_in_box: List[str] = []
                    codes_in_box: List[str] = []
                    for item in global_items:
                        cx, cy = item["center"]
                        if not (ex1 <= cx <= ex2 and ey1 <= cy <= ey2):
                            continue
                        txt = str(item.get("text", "")).strip()
                        if not txt:
                            continue
                        texts_in_box.append(txt)
                        codes_in_box.extend(_extract_codes_from_text(txt, code_re))

                    if codes_in_box:
                        uniq_codes = sorted(list({c for c in codes_in_box if c}))
                        die_codes[i] = uniq_codes
                    if texts_in_box and i not in die_texts:
                        # keep stable concise text for UI
                        uniq_texts = sorted(list({t for t in texts_in_box if t}))
                        die_texts[i] = " | ".join(uniq_texts)

        to1 = cv2.getTickCount()
        detections = []
        for i, (_area, class_id, class_name, conf, x1, y1, x2, y2) in enumerate(candidates):
            should_ocr = class_name.lower() == "die"
            if ocr_engine is not None and should_ocr:
                text = die_texts.get(i, "")
                if text:
                    ocr_texts = [text]
                    ocr_codes = die_codes.get(i, _extract_codes_from_text(text, code_re))
                else:
                    ocr_texts = []
                    ocr_codes = []
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

def process_images_from_camera(image_folder=None, pattern=CAPTURE_PATTERN):
    """
    澶勭悊娴峰悍鐩告満鎷嶆憚鐨勫浘鍍?
    Args:
        image_folder: 鍥惧儚鎵€鍦ㄦ枃浠跺す锛岄粯璁や负褰撳墠鐩綍
        pattern: 鍖归厤妯″紡锛岄粯璁や负capture_clear_*.jpg
    """
    if image_folder is None:
        image_folder = IMAGE_FOLDER
    
    with open(OUTPUT_LOG, "a", encoding="utf-8") as log_file:
        def log_line(message=""):
            print(message)
            log_file.write(f"{message}\n")

        log_line("===== 閰嶄欢鏂囧瓧璇嗗埆锛堟捣搴风浉鏈鸿緭鍑猴級=====")
        log_line()

        # 鑾峰彇鎵€鏈夊尮閰嶆ā寮忕殑鍥剧墖鏂囦欢
        file_pattern = os.path.join(image_folder, pattern)
        image_files = sorted(glob.glob(file_pattern), key=os.path.getctime, reverse=True)
        
        if not image_files:
            log_line(f"[WARN] 鏈壘鍒板尮閰嶆ā寮?'{pattern}' 鐨勫浘鐗囨枃浠跺湪 {image_folder}")
            return

        # 閬嶅巻鍥剧墖锛堜粠鏈€鏂板紑濮嬶級
        for img_path in image_files:
            filename = os.path.basename(img_path)
            img = cv2.imread(img_path)
            if img is None:
                log_line(f"[WARN] 鍥剧墖璇诲彇澶辫触: {filename}锛屽凡璺宠繃")
                continue
            
            results = model(img_path, conf=CONF_THRESH, verbose=False)

            log_line(f"========== {filename} ==========")

            # 閫愪釜妫€娴嬫璇嗗埆鏂囧瓧
            for idx, box in enumerate(results[0].boxes):
                class_id = int(box.cls.item()) if box.cls is not None else -1
                cls_name = results[0].names.get(class_id, str(class_id)) if class_id >= 0 else "unknown"

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img.shape[1], x2)
                y2 = min(img.shape[0], y2)

                if x2 <= x1 or y2 <= y1:
                    log_line(f"[{cls_name} {idx+1}] 妫€娴嬫鏃犳晥锛屽凡璺宠繃")
                    continue

                crop = img[y1:y2, x1:x2]

                # OCR 璇嗗埆
                res = list(ocr.predict(crop))
                text = ""
                if res:
                    first = res[0]
                    if isinstance(first, dict) and first.get("rec_texts"):
                        text = " ".join(first["rec_texts"])
                    elif hasattr(first, "rec_texts") and getattr(first, "rec_texts"):
                        text = " ".join(first.rec_texts)
                    elif isinstance(first, list):
                        text = " ".join([line[1][0] for line in first if len(line) > 1 and len(line[1]) > 0])

                log_line(f"[{cls_name} {idx+1}] OCR text: {text.strip()}")

            log_line()

print(f"OCR log saved to: {OUTPUT_LOG}")

if __name__ == "__main__":
    # 濡傛灉鐩存帴杩愯姝よ剼鏈紝澶勭悊鏈€鏂扮殑鐩告満鎷嶆憚鍥惧儚
    process_images_from_camera()


