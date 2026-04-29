from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class DigitCandidate:
    digit: str
    score: float
    bbox: Tuple[int, int, int, int]
    digit_scores: Dict[str, float]


class MnistDigitOCR:
    LEGAL_CODES = ("12", "16", "20", "25", "26")
    FAST_PLATE_MODEL = "cct-xs-v2-global-model"
    REFERENCE_LAYOUT = {
        "top": ("26", "25"),
        "bottom": ("20", "16", "12"),
    }

    """
    Visual 5-class die recognizer based on real reference crops.
    """

    def __init__(self, use_paddle_fallback: bool = False, prefer_paddle: bool = False) -> None:
        self.prototype_bank: Dict[str, List[np.ndarray]] = {code: [] for code in self.LEGAL_CODES}
        self.prototype_feature_bank: Dict[str, Dict[str, np.ndarray]] = {}
        self.position_priors: Dict[str, Tuple[float, float]] = {}
        self.prototype_ready = False
        self.use_paddle_fallback = bool(use_paddle_fallback)
        self.prefer_paddle = bool(prefer_paddle)
        self.rec_engine = None
        self._paddle_init_attempted = False
        self._paddle_init_done = threading.Event()
        self._paddle_init_lock = threading.Lock()
        self.paddle_error = ""

    def warmup(self) -> None:
        dummy = np.zeros((64, 128, 3), dtype=np.uint8)
        _ = self.recognize_die_code(dummy, expected_codes=None)

    def bootstrap_from_reference(self, reference_image_path: Path, model, conf_threshold: float) -> bool:
        ref_path = Path(reference_image_path)
        if (not ref_path.exists()) or model is None:
            return False
        img = cv2.imread(str(ref_path))
        if img is None:
            return False

        results = model(img, conf=float(conf_threshold), verbose=False, imgsz=960)
        result = results[0]
        names = result.names if hasattr(result, "names") else {}
        die_rows = []
        for box in result.boxes:
            class_id = int(box.cls.item()) if box.cls is not None else -1
            class_name = str(names.get(class_id, class_id)).lower()
            if class_name != "die":
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if x2 <= x1 or y2 <= y1:
                continue
            area = (x2 - x1) * (y2 - y1)
            if area < 500:
                continue
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            die_rows.append((cy, cx, x1, y1, x2, y2))

        if len(die_rows) < 5:
            return False

        die_rows = sorted(die_rows, key=lambda item: (item[0], item[1]))[:5]
        top = sorted(die_rows[:2], key=lambda item: item[1])
        bottom = sorted(die_rows[2:5], key=lambda item: item[1])
        labeled = list(zip(self.REFERENCE_LAYOUT["top"], top)) + list(zip(self.REFERENCE_LAYOUT["bottom"], bottom))

        bank: Dict[str, List[np.ndarray]] = {code: [] for code in self.LEGAL_CODES}
        h, w = img.shape[:2]
        position_priors: Dict[str, Tuple[float, float]] = {}
        for code, (_, _, x1, y1, x2, y2) in labeled:
            crop = self._pad_and_crop(img, x1, y1, x2, y2, pad_ratio=0.18)
            for mask in self._build_text_mask_variants(crop):
                bank[code].append(mask)
            position_priors[code] = (((x1 + x2) * 0.5) / max(1.0, float(w)), ((y1 + y2) * 0.5) / max(1.0, float(h)))

        ready = all(bank[code] for code in self.LEGAL_CODES)
        if ready:
            self.prototype_bank = bank
            self.prototype_feature_bank = {
                code: self._build_feature_bank(refs)
                for code, refs in bank.items()
            }
            self.position_priors = position_priors
            self.prototype_ready = True
        return ready

    def recognize_die_code(
        self,
        crop: np.ndarray,
        expected_codes: Optional[Sequence[str]] = None,
    ) -> Dict[str, List[str]]:
        if crop is None or getattr(crop, "size", 0) == 0:
            return {"texts": [], "codes": []}

        if self.prefer_paddle:
            ocr_text, ocr_code = self._recognize_with_fast_plate(crop)
            if ocr_code:
                return {
                    "texts": [ocr_text, f"TH{ocr_code}"] if ocr_text else [ocr_code, f"TH{ocr_code}"],
                    "codes": [f"TH{ocr_code}"],
                    "code_scores": {},
                    "engine": "fast-plate-ocr",
                }

        if not self.prototype_ready:
            return {"texts": [], "codes": [], "code_scores": {}}

        score_map = {code: -1.0 for code in self.LEGAL_CODES}
        for mask in self._build_text_mask_variants(crop):
            mask_feature = self._mask_feature(mask)
            for code, refs in self.prototype_bank.items():
                ref_features = self.prototype_feature_bank.get(code)
                if ref_features:
                    score = self._best_feature_similarity(mask_feature, ref_features)
                else:
                    score = max((self._mask_similarity(mask, ref) for ref in refs), default=-1.0)
                if score > score_map[code]:
                    score_map[code] = score

        best_code = max(score_map.items(), key=lambda item: item[1])[0]
        best_score = score_map[best_code]
        second_score = max([score for code, score in score_map.items() if code != best_code] or [-1.0])
        is_uncertain = best_score < 0.52 or (second_score >= 0.0 and (best_score - second_score) < 0.015)
        if is_uncertain and self.use_paddle_fallback:
            ocr_text, ocr_code = self._recognize_with_fast_plate(crop)
            if ocr_code:
                return {
                    "texts": [ocr_text, f"TH{ocr_code}"] if ocr_text else [ocr_code, f"TH{ocr_code}"],
                    "codes": [f"TH{ocr_code}"],
                    "code_scores": score_map,
                }
        if is_uncertain and best_score < 0.36:
            return {"texts": [], "codes": [], "code_scores": score_map}

        return {
            "texts": [best_code, f"TH{best_code}"],
            "codes": [f"TH{best_code}"],
            "code_scores": score_map,
        }

    def _resize_keep_aspect(self, img: np.ndarray, max_side: int) -> np.ndarray:
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest <= max_side:
            return img
        scale = max_side / float(longest)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)

    def _pad_and_crop(
        self,
        img: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        pad_ratio: float = 0.18,
    ) -> np.ndarray:
        h, w = img.shape[:2]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        px = int(round(bw * pad_ratio))
        py = int(round(bh * pad_ratio))
        xx1 = max(0, x1 - px)
        yy1 = max(0, y1 - py)
        xx2 = min(w, x2 + px)
        yy2 = min(h, y2 + py)
        return img[yy1:yy2, xx1:xx2]

    def _binarize_variants(self, crop: np.ndarray) -> List[np.ndarray]:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
        blur = cv2.GaussianBlur(clahe, (3, 3), 0)
        sharp = cv2.addWeighted(clahe, 1.7, blur, -0.7, 0)

        binaries: List[np.ndarray] = []
        for src in (clahe, sharp):
            _, inv = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            binaries.append(inv)
        return binaries

    def _build_text_mask_variants(self, crop: np.ndarray) -> List[np.ndarray]:
        base = self._resize_keep_aspect(crop, max_side=196)
        variants = []
        text_rois = self._extract_text_rois(base)
        for roi in text_rois:
            for img in [roi, cv2.rotate(roi, cv2.ROTATE_180)]:
                for binary in self._binarize_variants(img):
                    mask = self._normalize_text_mask(binary)
                    if mask is not None:
                        variants.append(mask)
        return variants

    def _extract_text_rois(self, crop: np.ndarray) -> List[np.ndarray]:
        h, w = crop.shape[:2]
        if h < 16 or w < 16:
            return [crop]
        rois = [
            crop[int(h * 0.56): min(h, int(h * 0.94)), 0:w],
            crop[int(h * 0.60): min(h, int(h * 0.96)), 0:int(w * 0.78)],
            crop[int(h * 0.48): min(h, int(h * 0.88)), int(w * 0.08):int(w * 0.92)],
        ]
        out = []
        for roi in rois:
            if roi is None or getattr(roi, "size", 0) == 0:
                continue
            out.append(roi)
        return out

    def _normalize_text_mask(self, binary: np.ndarray) -> Optional[np.ndarray]:
        coords = cv2.findNonZero(binary)
        if coords is None:
            return None
        x, y, w, h = cv2.boundingRect(coords)
        if w <= 0 or h <= 0:
            return None
        roi = binary[y:y + h, x:x + w]
        fill_ratio = float((roi > 0).mean())
        if fill_ratio < 0.01:
            return None
        canvas = np.zeros((48, 128), dtype=np.uint8)
        scale = min(118.0 / float(max(1, w)), 38.0 / float(max(1, h)))
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        resized = cv2.resize(
            roi,
            (nw, nh),
            interpolation=cv2.INTER_AREA if max(w, h) > max(nw, nh) else cv2.INTER_CUBIC,
        )
        ox = (128 - nw) // 2
        oy = (48 - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = resized
        kernel = np.ones((2, 2), np.uint8)
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)
        return (canvas > 0).astype(np.uint8)

    def _mask_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        af = self._mask_feature(a)
        bf = self._mask_feature(b)
        inter = float(np.logical_and(af["flat"] > 0, bf["flat"] > 0).sum())
        union = float(af["count"] + bf["count"] - inter)
        iou = inter / union if union > 0 else 0.0
        xor_ratio = float((af["count"] + bf["count"] - 2.0 * inter) / max(1, af["size"]))
        px = self._cosine_similarity(af["proj_x"], bf["proj_x"])
        py = self._cosine_similarity(af["proj_y"], bf["proj_y"])
        return iou * 0.52 + px * 0.24 + py * 0.24 - xor_ratio * 0.12

    def _mask_feature(self, mask: np.ndarray) -> Dict[str, Any]:
        binary = (mask > 0).astype(np.uint8)
        return {
            "flat": binary.reshape(-1),
            "count": float(binary.sum()),
            "size": int(binary.size),
            "proj_x": binary.sum(axis=0).astype(np.float32),
            "proj_y": binary.sum(axis=1).astype(np.float32),
        }

    def _build_feature_bank(self, refs: List[np.ndarray]) -> Dict[str, np.ndarray]:
        features = [self._mask_feature(ref) for ref in refs]
        return {
            "flat": np.stack([f["flat"] for f in features]).astype(np.uint8),
            "count": np.asarray([f["count"] for f in features], dtype=np.float32),
            "proj_x": np.stack([f["proj_x"] for f in features]).astype(np.float32),
            "proj_y": np.stack([f["proj_y"] for f in features]).astype(np.float32),
            "proj_x_norm": np.asarray(
                [max(float(np.linalg.norm(f["proj_x"])), 1e-6) for f in features],
                dtype=np.float32,
            ),
            "proj_y_norm": np.asarray(
                [max(float(np.linalg.norm(f["proj_y"])), 1e-6) for f in features],
                dtype=np.float32,
            ),
        }

    def _best_feature_similarity(self, mask_feature: Dict[str, Any], refs: Dict[str, np.ndarray]) -> float:
        ref_flat = refs["flat"]
        cand_flat = mask_feature["flat"]
        inter = np.logical_and(ref_flat, cand_flat).sum(axis=1).astype(np.float32)
        cand_count = float(mask_feature["count"])
        union = refs["count"] + cand_count - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        xor_ratio = (refs["count"] + cand_count - 2.0 * inter) / float(max(1, int(mask_feature["size"])))

        px_norm = max(float(np.linalg.norm(mask_feature["proj_x"])), 1e-6)
        py_norm = max(float(np.linalg.norm(mask_feature["proj_y"])), 1e-6)
        px = (refs["proj_x"] @ mask_feature["proj_x"]) / (refs["proj_x_norm"] * px_norm)
        py = (refs["proj_y"] @ mask_feature["proj_y"]) / (refs["proj_y_norm"] * py_norm)
        scores = iou * 0.52 + px * 0.24 + py * 0.24 - xor_ratio * 0.12
        return float(scores.max()) if scores.size else -1.0

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _recognize_with_fast_plate(self, crop: np.ndarray) -> Tuple[str, Optional[str]]:
        if not self.ensure_paddle_engine():
            return "", None

        alias_to_code = {
            "12": "12",
            "21": "12",
            "16": "16",
            "61": "16",
            "20": "20",
            "02": "20",
            "25": "25",
            "52": "25",
            "26": "26",
            "62": "26",
        }
        candidates = self._extract_text_rois(crop)
        for roi in candidates:
            for img in [roi, cv2.rotate(roi, cv2.ROTATE_180)]:
                try:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img
                    outs = list(self.rec_engine.run(rgb, return_confidence=False))
                except Exception:
                    outs = []
                texts = []
                for out in outs:
                    if hasattr(out, "plate"):
                        text = str(out.plate).strip()
                        if text:
                            texts.append(text)
                    elif isinstance(out, dict):
                        text = str(out.get("plate", "")).strip()
                        if text:
                            texts.append(text)
                for text in texts:
                    digits = "".join(ch for ch in text if ch.isdigit())
                    if len(digits) < 2:
                        continue
                    for i in range(len(digits) - 1):
                        pair = digits[i:i + 2]
                        if pair in alias_to_code:
                            return pair, alias_to_code[pair]
        return "", None

    def ensure_paddle_engine(self, timeout_s: float = 2.0) -> bool:
        if self.rec_engine is not None:
            return True
        with self._paddle_init_lock:
            if not self._paddle_init_attempted:
                self._paddle_init_attempted = True
                self.paddle_error = "initializing"
                threading.Thread(target=self._init_paddle_engine, daemon=True).start()

        if not self._paddle_init_done.wait(timeout=max(0.05, float(timeout_s))):
            self.paddle_error = f"initialization_timeout_after_{float(timeout_s):.1f}s"
            return False
        return self.rec_engine is not None

    def _init_paddle_engine(self) -> None:
        try:
            from fast_plate_ocr import LicensePlateRecognizer
            self.rec_engine = LicensePlateRecognizer(self.FAST_PLATE_MODEL, device="cpu")
            self.paddle_error = ""
        except Exception as e:
            self.paddle_error = str(e)
            self.rec_engine = None
        finally:
            self._paddle_init_done.set()
