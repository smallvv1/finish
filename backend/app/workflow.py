#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import datetime as dt
import glob
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from paddleocr import PaddleOCR
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def load_config(config_path: Path) -> Dict:
    # Accept both UTF-8 and UTF-8 with BOM.
    with config_path.open("r", encoding="utf-8-sig") as f:
        cfg = json.load(f)
    return cfg


def find_images(image_dir: Path, pattern: str, limit: Optional[int] = None) -> List[Path]:
    files = sorted(
        (Path(p) for p in glob.glob(str(image_dir / pattern))),
        key=lambda p: p.stat().st_ctime,
        reverse=True,
    )
    if limit and limit > 0:
        return files[:limit]
    return files


def parse_legacy_ocr_with_score(legacy_result, score_thresh: float) -> List[str]:
    texts: List[str] = []
    if not legacy_result:
        return texts

    lines = legacy_result[0] if isinstance(legacy_result, list) and legacy_result else []
    for line in lines:
        if (
            isinstance(line, (list, tuple))
            and len(line) > 1
            and isinstance(line[1], (list, tuple))
            and len(line[1]) > 1
        ):
            text = str(line[1][0]).strip()
            score = float(line[1][1])
            if text and score >= score_thresh:
                texts.append(text)
    return texts


def extract_texts(ocr_engine: PaddleOCR, image, score_thresh: float) -> List[str]:
    texts: List[str] = []

    if hasattr(ocr_engine, "predict"):
        result = list(ocr_engine.predict(image))
        if result:
            first = result[0]
            if isinstance(first, dict):
                rec_texts = first.get("rec_texts") or []
                texts.extend([str(item).strip() for item in rec_texts if str(item).strip()])
                return texts
            if hasattr(first, "rec_texts") and getattr(first, "rec_texts"):
                texts.extend([str(item).strip() for item in first.rec_texts if str(item).strip()])
                return texts

    legacy_result = ocr_engine.ocr(image, cls=True)
    texts.extend(parse_legacy_ocr_with_score(legacy_result, score_thresh))
    return texts


def extract_codes(texts: List[str], code_pattern: str) -> List[str]:
    codes: List[str] = []
    pattern = re.compile(code_pattern, flags=re.IGNORECASE)
    for txt in texts:
        found = pattern.findall(txt)
        if not found:
            continue
        if isinstance(found[0], tuple):
            found = ["".join(x) for x in found]
        for code in found:
            code_norm = str(code).strip().upper()
            if code_norm:
                codes.append(code_norm)
    return list(sorted(set(codes)))


def run_capture_step(capture_cfg: Dict, mode: str, count: int) -> None:
    script_path = _resolve_path(capture_cfg["script"])
    if not script_path.exists():
        raise FileNotFoundError(f"Capture script not found: {script_path}")

    python_exec = capture_cfg.get("python_executable") or sys.executable
    cmd = [python_exec, str(script_path), "--mode", mode, "--count", str(count)]
    print("[INFO] Running capture step:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def detect_and_ocr(
    image_paths: List[Path],
    model: YOLO,
    ocr_engine: Optional[PaddleOCR],
    conf_threshold: float,
    class_conf_thresholds: Dict[str, float],
    score_thresh: float,
    code_pattern: str,
    annotated_dir: Path,
) -> List[Dict]:
    reports: List[Dict] = []
    annotated_dir.mkdir(parents=True, exist_ok=True)

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

        results = model(str(image_path), conf=conf_threshold, verbose=False)
        result = results[0]
        names = result.names if hasattr(result, "names") else {}
        detections = []

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

            crop = img[y1:y2, x1:x2]
            if ocr_engine is not None:
                ocr_texts = extract_texts(ocr_engine, crop, score_thresh)
                ocr_codes = extract_codes(ocr_texts, code_pattern)
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


def evaluate_missing(image_reports: List[Dict], expected_tools: List[Dict]) -> Dict:
    observed_counts = defaultdict(int)
    observed_codes = defaultdict(set)

    for report in image_reports:
        if report.get("status") != "ok":
            continue
        for det in report.get("detections", []):
            class_name = det.get("class_name", "unknown")
            observed_counts[class_name] += 1
            for code in det.get("ocr_codes", []):
                observed_codes[class_name].add(code)

    items = []
    total_missing_items = 0
    total_missing_codes = 0

    for rule in expected_tools:
        class_name = str(rule.get("class_name", "")).strip()
        required_count = int(rule.get("required_count", 0))
        required_codes = [str(x).strip().upper() for x in rule.get("required_codes", []) if str(x).strip()]

        found_count = int(observed_counts.get(class_name, 0))
        missing_count = max(required_count - found_count, 0)

        found_codes = sorted(list(observed_codes.get(class_name, set())))
        missing_codes = [code for code in required_codes if code not in observed_codes.get(class_name, set())]

        total_missing_items += missing_count
        total_missing_codes += len(missing_codes)

        items.append(
            {
                "class_name": class_name,
                "required_count": required_count,
                "found_count": found_count,
                "missing_count": missing_count,
                "required_codes": required_codes,
                "found_codes": found_codes,
                "missing_codes": missing_codes,
            }
        )

    overall_status = "ok" if total_missing_items == 0 and total_missing_codes == 0 else "missing_detected"
    return {
        "status": overall_status,
        "total_missing_items": total_missing_items,
        "total_missing_codes": total_missing_codes,
        "details": items,
    }


def write_report(report_dir: Path, report: Dict) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"report_{ts}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report_path


def write_summary_text(report_dir: Path, report: Dict) -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = report_dir / f"summary_{ts}.txt"
    lines = []
    lines.append(f"Timestamp: {report.get('timestamp')}")
    lines.append(f"Overall status: {report['missing_evaluation']['status']}")
    lines.append(f"Images processed: {report.get('image_count', 0)}")
    lines.append("")
    lines.append("Missing details:")
    for item in report["missing_evaluation"]["details"]:
        lines.append(
            f"- {item['class_name']}: required={item['required_count']}, found={item['found_count']}, "
            f"missing={item['missing_count']}, missing_codes={item['missing_codes']}"
        )
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return txt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Toolbox capture + detect + OCR + missing evaluation workflow")
    parser.add_argument("--config", default="config/toolbox_workflow.json", help="Config file path")
    parser.add_argument("--skip-capture", action="store_true", help="Skip camera capture step")
    parser.add_argument("--mode", default=None, help="Capture mode override")
    parser.add_argument("--count", type=int, default=None, help="Capture count override")
    parser.add_argument("--limit-images", type=int, default=None, help="Process latest N matched images")
    args = parser.parse_args()

    cfg_path = _resolve_path(args.config)
    cfg = load_config(cfg_path)

    os.environ.setdefault("FLAGS_enable_pir_api", "0")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")

    model_path = _resolve_path(cfg["model"]["path"])
    image_dir = _resolve_path(cfg["input"]["image_dir"])
    image_pattern = cfg["input"].get("image_pattern", "capture_clear_*.jpg")

    conf_threshold = float(cfg["model"].get("conf_threshold", 0.4))
    class_conf_cfg = cfg["model"].get("class_conf_thresholds", {})
    class_conf_thresholds: Dict[str, float] = {}
    if isinstance(class_conf_cfg, dict):
        for k, v in class_conf_cfg.items():
            try:
                class_conf_thresholds[str(k).strip().lower()] = float(v)
            except Exception:
                continue
    ocr_cfg = cfg.get("ocr", {})
    ocr_enabled = bool(ocr_cfg.get("enabled", True))
    allow_ocr_failure = bool(ocr_cfg.get("allow_failure", True))
    ocr_lang = str(ocr_cfg.get("lang", "en"))
    ocr_score_thresh = float(ocr_cfg.get("score_threshold", 0.5))
    code_pattern = str(cfg["rules"].get("code_pattern", r"TH\d{2}"))
    expected_tools = cfg["rules"].get("expected_tools", [])

    runtime_root = _resolve_path(cfg["output"].get("runtime_dir", "runtime"))
    report_dir = runtime_root / "reports"
    annotated_dir = runtime_root / "annotated"
    tmp_dir = runtime_root / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(tmp_dir)
    os.environ["TMP"] = str(tmp_dir)

    if not args.skip_capture:
        capture_cfg = cfg.get("capture", {})
        capture_mode = args.mode or capture_cfg.get("mode", "fast")
        capture_count = args.count if args.count is not None else int(capture_cfg.get("count", 1))
        run_capture_step(capture_cfg, capture_mode, capture_count)

    limit_images = args.limit_images if args.limit_images is not None else int(cfg["input"].get("default_limit_images", 5))
    image_paths = find_images(image_dir, image_pattern, limit=limit_images)
    if not image_paths:
        raise FileNotFoundError(f"No images found: dir={image_dir}, pattern={image_pattern}")

    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(str(model_path))
    ocr_engine: Optional[PaddleOCR] = None
    ocr_status = "disabled"
    if ocr_enabled:
        try:
            print(f"[INFO] Initializing OCR: lang={ocr_lang}")
            ocr_engine = PaddleOCR(use_textline_orientation=True, lang=ocr_lang, enable_mkldnn=False)
            ocr_status = "ready"
        except Exception as e:
            ocr_status = f"failed: {e}"
            print(f"[WARN] OCR init failed: {e}")
            if not allow_ocr_failure:
                raise

    image_reports = detect_and_ocr(
        image_paths=image_paths,
        model=model,
        ocr_engine=ocr_engine,
        conf_threshold=conf_threshold,
        class_conf_thresholds=class_conf_thresholds,
        score_thresh=ocr_score_thresh,
        code_pattern=code_pattern,
        annotated_dir=annotated_dir,
    )

    missing_result = evaluate_missing(image_reports, expected_tools)
    report = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "config_path": str(cfg_path),
        "model_path": str(model_path),
        "ocr_status": ocr_status,
        "image_count": len(image_paths),
        "image_reports": image_reports,
        "missing_evaluation": missing_result,
    }

    report_path = write_report(report_dir, report)
    summary_path = write_summary_text(report_dir, report)

    print("[INFO] Workflow finished")
    print(f"[INFO] Report JSON: {report_path}")
    print(f"[INFO] Summary TXT: {summary_path}")
    print(f"[INFO] Overall status: {missing_result['status']}")


if __name__ == "__main__":
    main()
