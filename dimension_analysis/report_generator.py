"""
Inspection Report Generator — Stage 8.

PRE-CLAMP PROTOTYPE mode:
  - All displayed values are in PIXELS (image-domain measurements).
  - No PASS/FAIL evaluation is shown.
  - No tolerance values are shown.
  - px→mm conversion code is preserved in comments for future clamp integration.

Output artefacts:
  reports/<stem>_report.csv   — tabular, pixel columns
  reports/<stem>_report.json  — structured, pixel columns
  reports/<stem>_report.png   — composite inspection image (px values, no PASS/FAIL)
"""

import csv
import json
import logging
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from dimension_analysis.feature_matcher import MatchedPair
from dimension_analysis.tolerance import ToleranceResult

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")

# ── Palette ──────────────────────────────────────────────────────────────────
# BGR colour values (OpenCV uses BGR not RGB)
_C_NEUTRAL = (50,  150, 220)   # sky blue in BGR — neutral overlay (no PASS/FAIL)
_C_TEXT    = (255, 255, 255)
_C_DARK    = (18,   18,  18)
_C_PANEL   = (28,   28,  28)
_C_GRID    = (55,   55,  55)
_C_CALLOUT = (220, 220, 220)

_FONT  = cv2.FONT_HERSHEY_SIMPLEX
_FONT2 = cv2.FONT_HERSHEY_DUPLEX
_FS    = 0.36
_FM    = 0.44
_FL    = 0.58
_TH    = 1
_TH2   = 2

_IMG_TARGET_W = 900


def _ensure_reports_dir() -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR


def _mm_to_px(mm_val: float, scale_px_per_mm: float) -> float:
    """Convert a mm measurement to pixels using the given scale."""
    return mm_val * scale_px_per_mm


def _build_record(tr: ToleranceResult, present: bool, scale_px_per_mm: float) -> dict:
    """
    Build a CSV/JSON record in pixel units.

    Values from MeasuredFeature / ToleranceResult are already in px —
    no conversion needed here.

    # ── PRESERVED: mm output for future clamp-phase reactivation ──────────
    # When real-world scale is available, divide px values by scale_px_per_mm:
    # return {
    #     "feature_name":     tr.label,
    #     "feature_present":  "YES" if present else "NO",
    #     "cad_size_mm":      round(tr.cad_dimension_mm / scale_px_per_mm, 4),
    #     "measured_size_mm": round(tr.measured_dimension_mm / scale_px_per_mm, 4),
    #     "deviation_mm":     round(tr.deviation_mm / scale_px_per_mm, 4),
    #     "tolerance_mm":     round(tr.tolerance_mm, 4),
    #     "status":           tr.status,
    #     "unit":             "mm",
    # }
    """
    return {
        "feature_name":     tr.label,
        "feature_present":  "YES" if present else "NO",
        "cad_size_px":      round(tr.cad_dimension_mm, 2),
        "measured_size_px": round(tr.measured_dimension_mm, 2),
        "deviation_px":     round(tr.deviation_mm, 2),
        # tolerance and status omitted — evaluation disabled
    }


def _write_csv(records: list[dict], path: Path) -> None:
    if not records:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _write_json(report: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _put(img, text, x, y, color=_C_TEXT, fs=_FM, th=_TH, font=_FONT):
    cv2.putText(img, text, (int(x), int(y)), font, fs, color, th, cv2.LINE_AA)


def _callout_box(img, lines: list[tuple[str, tuple]], cx, cy, color):
    fs, th_val = _FS, _TH
    pad = 5
    line_h = 16
    widths = []
    for txt, _ in lines:
        (tw, _), _ = cv2.getTextSize(txt, _FONT, fs, th_val)
        widths.append(tw)
    box_w = max(widths) + pad * 2 + 2
    box_h = len(lines) * line_h + pad * 2

    H, W = img.shape[:2]
    bx = min(cx + 10, W - box_w - 2)
    by = max(2, min(cy - box_h // 2, H - box_h - 2))

    cv2.line(img, (cx, cy), (bx, by + box_h // 2), color, 1, cv2.LINE_AA)
    cv2.rectangle(img, (bx, by), (bx + box_w, by + box_h), _C_DARK, -1)
    cv2.rectangle(img, (bx, by), (bx + box_w, by + box_h), color, 1, cv2.LINE_AA)

    for i, (txt, tcol) in enumerate(lines):
        cv2.putText(img, txt,
                    (bx + pad, by + pad + line_h * (i + 1) - 2),
                    _FONT, fs, tcol, th_val, cv2.LINE_AA)


def _draw_index_dot(img, idx: int, cx: int, cy: int, color):
    r = 9
    cv2.circle(img, (cx, cy), r, color, -1)
    cv2.circle(img, (cx, cy), r, _C_DARK, 1, cv2.LINE_AA)
    label = str(idx)
    (tw, th_), _ = cv2.getTextSize(label, _FONT, 0.32, 1)
    cv2.putText(img, label, (cx - tw // 2, cy + th_ // 2),
                _FONT, 0.32, _C_DARK, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Visual report builder
# ---------------------------------------------------------------------------

def _build_visual_report(
    real_gray: np.ndarray,
    matched_pairs: list[MatchedPair],
    tolerance_results: list[ToleranceResult],
    image_stem: str,
    identified_as: str,
    alignment_score: float,
    coverage: float,
    scale_px_per_mm: float,
) -> np.ndarray:

    # ── 1. Photo panel ───────────────────────────────────────────────────
    photo = cv2.cvtColor(real_gray, cv2.COLOR_GRAY2BGR)
    H0, W0 = photo.shape[:2]
    scale_vis = _IMG_TARGET_W / W0
    new_w = _IMG_TARGET_W
    new_h = int(H0 * scale_vis)
    photo = cv2.resize(photo, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pair_by_label: dict[str, MatchedPair] = {p.label: p for p in matched_pairs}

    def _find_pair(tr: ToleranceResult):
        p = pair_by_label.get(tr.label)
        if p:
            return p
        base = tr.label.rsplit("_", 1)[0]
        return pair_by_label.get(base)

    feature_index: dict[str, int] = {}
    idx_counter = 1
    for p in matched_pairs:
        feature_index[p.label] = idx_counter
        idx_counter += 1

    tol_by_pair: dict[str, list[ToleranceResult]] = defaultdict(list)
    for tr in tolerance_results:
        p = _find_pair(tr)
        if p:
            tol_by_pair[p.label].append(tr)

    drawn_locations: set[str] = set()

    # ── 2a. Build a map from center_bore ToleranceResult → image position ──
    # The center bore pair has label "hole #N" but its ToleranceResult has
    # label "center_bore". We find it by matching cad_value_mm*2 ≈ center_bore.
    center_bore_pos_px: tuple | None = None
    center_bore_r_px: float | None   = None
    for p in matched_pairs:
        if p.feature_type == "circle":
            for tr in tolerance_results:
                if tr.label == "center_bore":
                    # The pair that contributed center_bore had cad_value_mm ≈ tr.cad_dimension_mm/2
                    s_p = p.scale_px_per_mm if p.scale_px_per_mm > 0 else (scale_px_per_mm if scale_px_per_mm > 0 else 1.0)
                    if abs(float(p.cad_value_mm) * 2.0 * s_p - tr.cad_dimension_mm) < 2.0:
                        center_bore_pos_px = p.image_pos_px
                        center_bore_r_px   = float(p.image_value_px)
                        break
            if center_bore_pos_px:
                break

    # ── 2. Overlays — draw ALL matched pairs (detection boxes always shown) ──
    for pair in matched_pairs:
        color = _C_NEUTRAL   # single neutral colour — no green/red

        cx = int(round(pair.image_pos_px[0] * scale_vis))
        cy = int(round(pair.image_pos_px[1] * scale_vis))
        loc_key = f"{cx},{cy}"
        if loc_key in drawn_locations:
            continue
        drawn_locations.add(loc_key)

        idx = feature_index[pair.label]
        s   = scale_px_per_mm if scale_px_per_mm > 0 else 1.0

        # Get tolerance rows for this pair (may be empty if measurement was skipped)
        trs = tol_by_pair.get(pair.label, [])

        if pair.feature_type == "circle":
            r = max(3, int(round(float(pair.image_value_px) * scale_vis)))

            ov = photo.copy()
            cv2.circle(ov, (cx, cy), r, color, -1)
            cv2.addWeighted(ov, 0.22, photo, 0.78, 0, photo)
            cv2.circle(photo, (cx, cy), r,     color, 2, cv2.LINE_AA)
            cv2.circle(photo, (cx, cy), r + 4, color, 1, cv2.LINE_AA)

            # Dashed expected CAD circle (cad_value_mm already scaled to px in Stage 6)
            cad_r_vis = max(3, int(round(float(pair.cad_value_mm) * s * scale_vis)))
            for seg in range(0, 360, 20):
                cv2.ellipse(photo, (cx, cy), (cad_r_vis, cad_r_vis),
                            0, seg, seg + 10, (180, 180, 60), 1, cv2.LINE_AA)

            # Callout — values are already in px
            if trs:
                meas_r_px = float(pair.image_value_px)
                cad_r_px  = float(pair.cad_value_mm) * s
                dev_r_px  = meas_r_px - cad_r_px
                callout_lines = [
                    (f"#{idx} {pair.label}", _C_CALLOUT),
                    (f"CAD  r={cad_r_px:.1f}px", (160, 160, 160)),
                    (f"Meas r={meas_r_px:.1f}px", _C_TEXT),
                    (f"Dev  {dev_r_px:+.1f}px", color),
                ]
                _callout_box(photo, callout_lines, cx, cy - r - 4, color)
            _draw_index_dot(photo, idx, cx, cy, color)

            # ── PRESERVED: mm callout for future clamp integration ────────
            # Divide by scale_px_per_mm to convert back to mm when available:
            # meas_r_mm = float(pair.image_value_px) / scale_px_per_mm
            # cad_r_mm  = float(pair.cad_value_mm)   / ... (depends on chain)
            # dev_r_mm  = meas_r_mm - cad_r_mm

        elif pair.feature_type == "rect":
            img_wh = pair.image_value_px
            hw = int(round(img_wh[0] * scale_vis / 2))
            hh = int(round(img_wh[1] * scale_vis / 2))
            x1, y1 = cx - hw, cy - hh
            x2, y2 = cx + hw, cy + hh

            ov = photo.copy()
            cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(ov, 0.22, photo, 0.78, 0, photo)
            cv2.rectangle(photo, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Dashed expected CAD rect
            cad_hw = int(round(float(pair.cad_value_mm[0]) * s * scale_vis / 2))
            cad_hh = int(round(float(pair.cad_value_mm[1]) * s * scale_vis / 2))
            for pts in [
                [(cx-cad_hw, cy-cad_hh), (cx+cad_hw, cy-cad_hh)],
                [(cx+cad_hw, cy-cad_hh), (cx+cad_hw, cy+cad_hh)],
                [(cx+cad_hw, cy+cad_hh), (cx-cad_hw, cy+cad_hh)],
                [(cx-cad_hw, cy+cad_hh), (cx-cad_hw, cy-cad_hh)],
            ]:
                p1, p2 = pts
                dist = int(math.hypot(p2[0]-p1[0], p2[1]-p1[1]))
                for d in range(0, dist, 10):
                    t0, t1 = d / dist, min((d + 5) / dist, 1.0)
                    cv2.line(photo,
                             (int(p1[0]+(p2[0]-p1[0])*t0), int(p1[1]+(p2[1]-p1[1])*t0)),
                             (int(p1[0]+(p2[0]-p1[0])*t1), int(p1[1]+(p2[1]-p1[1])*t1)),
                             (180, 180, 60), 1, cv2.LINE_AA)

            # Callout — values are already in px
            if trs:
                mw_px = float(img_wh[0])
                mh_px = float(img_wh[1])
                cw_px = float(pair.cad_value_mm[0]) * s
                ch_px = float(pair.cad_value_mm[1]) * s
                callout_lines = [
                    (f"#{idx} {pair.label}", _C_CALLOUT),
                    (f"CAD  {cw_px:.1f}x{ch_px:.1f}px", (160, 160, 160)),
                    (f"Meas {mw_px:.1f}x{mh_px:.1f}px", _C_TEXT),
                    (f"dW={mw_px-cw_px:+.1f} dH={mh_px-ch_px:+.1f}px", color),
                ]
                _callout_box(photo, callout_lines, cx + hw + 4, cy, color)
            _draw_index_dot(photo, idx, cx, cy, color)

            # ── PRESERVED: mm callout for future clamp integration ────────
            # Divide by scale_px_per_mm when real-world scale is available:
            # mw_mm = float(img_wh[0]) / scale_px_per_mm
            # mh_mm = float(img_wh[1]) / scale_px_per_mm

    # ── 2b. Draw center bore overlay if available ────────────────────────
    # center_bore has a ToleranceResult but no MatchedPair with that label.
    # Its image position is held by the pair that was reclassified (hole #19).
    if center_bore_pos_px is not None and center_bore_r_px is not None:
        cb_cx = int(round(center_bore_pos_px[0] * scale_vis))
        cb_cy = int(round(center_bore_pos_px[1] * scale_vis))
        cb_r  = max(3, int(round(center_bore_r_px * scale_vis)))
        cb_color = (0, 200, 255)   # amber/gold in BGR — distinct from bolt holes

        cb_tr = next((t for t in tolerance_results if t.label == "center_bore"), None)

        ov = photo.copy()
        cv2.circle(ov, (cb_cx, cb_cy), cb_r, cb_color, -1)
        cv2.addWeighted(ov, 0.28, photo, 0.72, 0, photo)
        cv2.circle(photo, (cb_cx, cb_cy), cb_r,     cb_color, 3, cv2.LINE_AA)
        cv2.circle(photo, (cb_cx, cb_cy), cb_r + 6, cb_color, 1, cv2.LINE_AA)

        # Cross-hair at centre
        arm = 12
        cv2.line(photo, (cb_cx - arm, cb_cy), (cb_cx + arm, cb_cy), cb_color, 1, cv2.LINE_AA)
        cv2.line(photo, (cb_cx, cb_cy - arm), (cb_cx, cb_cy + arm), cb_color, 1, cv2.LINE_AA)

        if cb_tr is not None:
            # Values already in px — no * s_cb needed
            cad_d_px  = cb_tr.cad_dimension_mm
            meas_d_px = cb_tr.measured_dimension_mm
            dev_d_px  = cb_tr.deviation_mm
            cb_lines = [
                ("CENTER BORE", _C_CALLOUT),
                (f"CAD  d={cad_d_px:.1f}px", (160, 160, 160)),
                (f"Meas d={meas_d_px:.1f}px", _C_TEXT),
                (f"Dev  {dev_d_px:+.1f}px", cb_color),
            ]
            _callout_box(photo, cb_lines, cb_cx + cb_r + 6, cb_cy, cb_color)

    # ── 3. Header bar — no PASS/FAIL ─────────────────────────────────────
    hdr_h = 52
    hdr = np.full((hdr_h, new_w, 3), _C_DARK, dtype=np.uint8)
    cv2.line(hdr, (0, hdr_h-1), (new_w, hdr_h-1), _C_GRID, 1)

    _put(hdr, f"INSPECTION  {image_stem.upper()}",   8, 20, _C_TEXT, _FL, _TH2)
    _put(hdr, f"Identified as: {identified_as}",     8, 40, (180,180,180), _FS)
    _put(hdr, f"Score:{alignment_score:.3f}",      380, 20, (180,180,180), _FM)
    _put(hdr, f"Cov:{coverage:.1%}",               510, 20, (180,180,180), _FM)
    _put(hdr, f"Scale:{scale_px_per_mm:.2f}px/mm", 620, 20, (180,180,180), _FM)
    _put(hdr, f"Units: px",                        780, 20, (180,180,180), _FS)
    _put(hdr, f"(pre-clamp prototype)",              8, 40, (140,140,140), _FS)

    # ── PRESERVED: PASS/FAIL header for future reactivation ──────────────
    # overall_str = "PASS" if fail_count == 0 else "FAIL"
    # ov_col = _C_PASS if overall_str == "PASS" else _C_FAIL
    # _put(hdr, f"{overall_str}  {pass_count}P/{fail_count}F", 380, 40, ov_col, _FM, _TH2)

    # ── 4. Legend panel — px values, no tolerance/status columns ─────────
    LEG_W  = 430
    ROW_H  = 20
    PAD    = 10
    HDR_ROW = 34

    C_IDX  = 28
    C_NAME = 170
    C_CAD  = 80
    C_MEAS = 80
    C_DEV  = 72

    rows = list(tolerance_results)
    n_total_rows = len(rows) + 4
    leg_h_content = HDR_ROW + n_total_rows * ROW_H + PAD * 3
    leg_h = max(new_h + hdr_h, leg_h_content)

    leg = np.full((leg_h, LEG_W, 3), _C_PANEL, dtype=np.uint8)
    cv2.line(leg, (0, 0), (0, leg_h), _C_GRID, 2)

    y = PAD + 16
    _put(leg, "DIMENSION REPORT  (pixels)", PAD, y, _C_TEXT, _FM, _TH2)
    y += 6
    cv2.line(leg, (PAD, y), (LEG_W - PAD, y), _C_GRID, 1)
    y += ROW_H

    x0 = PAD
    _put(leg, "#",       x0,                    y, (140,140,140), _FS)
    _put(leg, "Feature", x0+C_IDX,              y, (140,140,140), _FS)
    _put(leg, "CAD px",  x0+C_IDX+C_NAME,       y, (140,140,140), _FS)
    _put(leg, "Meas px", x0+C_IDX+C_NAME+C_CAD, y, (140,140,140), _FS)
    _put(leg, "Dev px",  x0+C_IDX+C_NAME+C_CAD+C_MEAS, y, (140,140,140), _FS)
    y += 4
    cv2.line(leg, (PAD, y), (LEG_W - PAD, y), _C_GRID, 1)
    y += ROW_H - 4

    s = scale_px_per_mm if scale_px_per_mm > 0 else 1.0

    for row_i, tr in enumerate(rows):
        if row_i % 2 == 0:
            cv2.rectangle(leg, (0, y - ROW_H + 4), (LEG_W, y + 4), (35,35,35), -1)

        p = pair_by_label.get(tr.label)
        if p is None:
            base = tr.label.rsplit("_", 1)[0]
            p = pair_by_label.get(base)
        idx_str = str(feature_index.get(p.label, "-")) if p else "-"

        name    = tr.label[:22]
        # Values are already in px — no * s needed
        cad_px  = f"{tr.cad_dimension_mm:.1f}"
        meas_px = f"{tr.measured_dimension_mm:.1f}"
        dev_px  = f"{tr.deviation_mm:+.1f}"

        # ── PRESERVED: mm display for future reactivation ─────────────
        # Divide by scale_px_per_mm when real-world scale is available:
        # cad_v  = f"{tr.cad_dimension_mm / s:.3f}mm"
        # meas_v = f"{tr.measured_dimension_mm / s:.3f}mm"
        # dev_v  = f"{tr.deviation_mm / s:+.3f}mm"

        _put(leg, idx_str,  x0,                          y, (160,160,160), _FS)
        _put(leg, name,     x0+C_IDX,                    y, _C_TEXT,       _FS)
        _put(leg, cad_px,   x0+C_IDX+C_NAME,             y, (160,160,160), _FS)
        _put(leg, meas_px,  x0+C_IDX+C_NAME+C_CAD,       y, _C_TEXT,       _FS)
        _put(leg, dev_px,   x0+C_IDX+C_NAME+C_CAD+C_MEAS,y, _C_NEUTRAL,    _FS)

        # ── PRESERVED: status column for future reactivation ──────────
        # tick = "OK" if tr.status == "PASS" else "FAIL"
        # color = _C_PASS if tr.status == "PASS" else _C_FAIL
        # _put(leg, tick, x0+C_IDX+C_NAME+C_CAD+C_MEAS+C_DEV, y, color, _FS)

        y += ROW_H

    y += 6
    cv2.line(leg, (PAD, y), (LEG_W - PAD, y), _C_GRID, 1)
    y += ROW_H
    _put(leg, f"Total features: {len(rows)}", PAD, y, _C_TEXT, _FM)
    y += ROW_H + 4
    _put(leg, "Pre-clamp prototype", PAD, y, (140,140,140), _FS)

    # ── PRESERVED: overall PASS/FAIL badge for future reactivation ────────
    # badge_text = f"  OVERALL: {overall_str}  "
    # badge_color = _C_PASS if overall_str == "PASS" else _C_FAIL
    # (bw, bh_), _ = cv2.getTextSize(badge_text, _FONT2, _FL, _TH2)
    # bx, by = PAD, y
    # cv2.rectangle(leg, (bx-4, by-bh_-4), (bx+bw+4, by+6), badge_color, -1)
    # cv2.putText(leg, badge_text, (bx, by), _FONT2, _FL, _C_DARK, _TH2, cv2.LINE_AA)

    # ── 5. Assemble ───────────────────────────────────────────────────────
    img_panel = np.vstack([hdr, photo])
    total_h = max(img_panel.shape[0], leg_h)
    if img_panel.shape[0] < total_h:
        pad = np.full((total_h - img_panel.shape[0], new_w, 3), _C_DARK, dtype=np.uint8)
        img_panel = np.vstack([img_panel, pad])
    if leg.shape[0] < total_h:
        pad = np.full((total_h - leg.shape[0], LEG_W, 3), _C_PANEL, dtype=np.uint8)
        leg = np.vstack([leg, pad])

    return np.hstack([img_panel, leg])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_reports(
    image_stem: str,
    identified_as: str,
    tolerance_results: list[ToleranceResult],
    matched_pairs: list[MatchedPair],
    scale_px_per_mm: float,
    alignment_score: float,
    coverage: float,
    strategy: str,
    real_gray=None,
    extra_metadata: dict | None = None,
) -> dict:
    out_dir = _ensure_reports_dir()

    s = scale_px_per_mm if scale_px_per_mm > 0 else 1.0
    matched_labels = {p.label for p in matched_pairs}

    records = []
    for tr in tolerance_results:
        present = tr.label in matched_labels or any(
            tr.label.startswith(p.label) for p in matched_pairs
        )
        records.append(_build_record(tr, present, s))

    total     = len(records)
    timestamp = datetime.now().isoformat(timespec="seconds")

    report = {
        "report_metadata": {
            "timestamp":       timestamp,
            "image":           image_stem,
            "identified_as":   identified_as,
            "alignment_score": round(alignment_score, 4),
            "coverage":        round(coverage, 4),
            "strategy":        strategy,
            "scale_px_per_mm": round(scale_px_per_mm, 4),
            "total_features":  total,
            "units":           "px",
            "evaluation":      "disabled (pre-clamp prototype)",
            # ── PRESERVED: pass/fail counts for future reactivation ────
            # "pass_count": pass_count,
            # "fail_count": fail_count,
            # "overall_status": overall,
        },
        "features": records,
    }
    if extra_metadata:
        report["extra"] = extra_metadata

    csv_path  = out_dir / f"{image_stem}_report.csv"
    json_path = out_dir / f"{image_stem}_report.json"
    png_path  = out_dir / f"{image_stem}_report.png"

    _write_csv(records, csv_path)
    _write_json(report, json_path)

    if real_gray is not None:
        try:
            vis = _build_visual_report(
                real_gray, matched_pairs, tolerance_results,
                image_stem, identified_as, alignment_score, coverage,
                scale_px_per_mm,
            )
            cv2.imwrite(str(png_path), vis)
        except Exception as exc:
            logger.warning(f"PNG failed: {exc}")
            png_path = None

    # Console output — px values, no PASS/FAIL
    SEP = "-" * 72
    print(f"\n{SEP}")
    print(f"  INSPECTION REPORT (px)  --  {image_stem}  ->  {identified_as}")
    print(SEP)
    print(f"  Timestamp : {timestamp}")
    print(f"  Score     : {alignment_score:.4f}   Coverage: {coverage:.1%}")
    print(f"  Scale ref : {scale_px_per_mm:.4f} px/mm")
    print(f"  Features  : {total}   (no PASS/FAIL — pre-clamp prototype)")
    print(SEP)
    print(f"  {'Feature':<28} {'Present':<8} {'CAD px':>8} {'Meas px':>8} {'Dev px':>8}")
    print(f"  {'-'*68}")

    for r in records:
        print(
            f"  {r['feature_name']:<28} "
            f"{r['feature_present']:<8} "
            f"{r['cad_size_px']:>8.1f} "
            f"{r['measured_size_px']:>8.1f} "
            f"{r['deviation_px']:>+8.1f}"
        )

    # ── PRESERVED: PASS/FAIL console summary for future reactivation ──────
    # print(SEP)
    # overall_tag = "[PASS]" if overall == "PASS" else "[FAIL]"
    # print(f"  OVERALL: {overall_tag}   ({pass_count}/{total} pass)")

    print(SEP)
    png_name = png_path.name if png_path else "(none)"
    print(f"  Saved: {csv_path.name}  |  {json_path.name}  |  {png_name}")
    print(f"{SEP}\n")

    return report
