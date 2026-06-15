"""
Feature Measurement — Stage 6.

All measurements are reported in PIXELS (image-domain).

For every matched feature the pipeline produces:
  cad_dimension_px   : CAD nominal scaled to image pixels
                       (cad_value_mm × total_scale)
  measured_px        : what was actually detected in the image (px)
  deviation_px       : measured_px − cad_dimension_px

No mm conversion is performed here. Real-world unit conversion requires
a physical calibration reference (clamp fixture) and is deferred to a
future phase.

ACTUAL IMAGE MEASUREMENT STRATEGY
----------------------------------
For circles  : a HoughCircles search is run in a small RoI around the
               projected CAD centre.  The fitted radius (px) is returned
               directly.  Skipped when Hough fails.
For rects    : contour bounding box in a local RoI around the projected
               centre.  Width and height in px returned directly.
For spacing  : pixel distance between Hough-detected hole centres.
               Skipped unless both holes were independently detected.
For PCD      : mean bolt-hole radius from Hough-detected centres (px).
For overall  : measured from the part mask contour bounding box (px).
"""

import logging
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from dimension_analysis.dxf_parser import CADFeatureSet
from dimension_analysis.feature_matcher import MatchedPair, _project_cad_to_image
from dimension_analysis.transform_estimator import TransformResult

logger = logging.getLogger(__name__)

# Search radius multiplier when looking for a circle in the image
_HOUGH_SEARCH_FACTOR = 3.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_circle_in_roi(
    real_gray: np.ndarray,
    cx_img_px: float,
    cy_img_px: float,
    expected_r_px: float,
) -> tuple[float, float, float] | None:
    """
    Fit HoughCircles in a local RoI around (cx_img_px, cy_img_px).
    Returns (cx_px, cy_px, radius_px) or None if Hough fails.
    """
    h, w = real_gray.shape
    search_r = int(math.ceil(expected_r_px * _HOUGH_SEARCH_FACTOR))
    x1 = max(0, int(cx_img_px) - search_r)
    y1 = max(0, int(cy_img_px) - search_r)
    x2 = min(w, int(cx_img_px) + search_r)
    y2 = min(h, int(cy_img_px) + search_r)

    roi = real_gray[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    min_r = max(2, int(expected_r_px * 0.5))
    max_r = max(min_r + 2, int(expected_r_px * 2.0))

    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(4, min_r),
        param1=50,
        param2=18,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is None:
        return None

    best_cx_px: float | None = None
    best_cy_px: float | None = None
    best_r_px: float | None = None
    best_dist = 1e9
    for cx_roi, cy_roi, r_roi in circles[0]:
        cx_abs = float(cx_roi + x1)
        cy_abs = float(cy_roi + y1)
        dist = math.hypot(cx_abs - cx_img_px, cy_abs - cy_img_px)
        if dist < best_dist:
            best_dist = dist
            best_cx_px = cx_abs
            best_cy_px = cy_abs
            best_r_px = float(r_roi)

    if best_r_px is None or best_cx_px is None or best_cy_px is None:
        return None

    # Return raw pixel values — no mm conversion
    return best_cx_px, best_cy_px, best_r_px


# def _detect_circle_in_roi_mm(...)  ← old mm-returning variant removed.
# Keeping this comment so git blame explains the change.

def _measure_circle_radius_px(
    real_gray: np.ndarray,
    cx_img_px: float,
    cy_img_px: float,
    expected_r_px: float,
) -> float | None:
    """Return measured radius in px, or None if Hough fails."""
    hit = _detect_circle_in_roi(real_gray, cx_img_px, cy_img_px, expected_r_px)
    return hit[2] if hit is not None else None


def _measure_overall_from_mask(
    real_gray: np.ndarray,
) -> tuple[float | None, float | None]:
    """
    Estimate overall width and height in PIXELS from the part bounding box.

    Tries both THRESH_BINARY and THRESH_BINARY_INV, picks the result whose
    largest contour area is between 5% and 85% of the image area — this
    rejects both the near-empty case and the full-background case.

    Returns (width_px, height_px), or (None, None) on failure.
    """
    blur = cv2.GaussianBlur(real_gray, (5, 5), 0)
    img_area = float(real_gray.shape[0] * real_gray.shape[1])

    def _try_thresh(flags: int):
        _, mask = cv2.threshold(blur, 0, 255, flags)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < img_area * 0.05 or area > img_area * 0.85:
            return None
        _, _, w_px, h_px = cv2.boundingRect(largest)
        # Return raw px — no /total_scale division
        return float(w_px), float(h_px)

    result = _try_thresh(cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if result is not None:
        return result

    result = _try_thresh(cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if result is not None:
        return result

    logger.warning("_measure_overall_from_mask: no valid part contour found in image")
    return None, None


# ---------------------------------------------------------------------------
# Public data structure
# ---------------------------------------------------------------------------

@dataclass
class MeasuredFeature:
    """One feature with CAD nominal, image-measured value, and deviation — all in PIXELS."""
    feature_type: str
    label: str
    cad_dimension_px: float       # CAD nominal in image pixels (cad_mm × total_scale)
    measured_dimension_px: float  # measured from image in pixels
    deviation_px: float           # measured_px − cad_px
    unit: str = "px"
    extra: dict = field(default_factory=dict)

    # ── Backward-compat aliases (old field names used mm suffix) ──────────
    # Kept so any external code referencing the old names still works.
    # They are always equal to the _px fields above.
    @property
    def cad_dimension_mm(self) -> float:
        return self.cad_dimension_px

    @property
    def measured_dimension_mm(self) -> float:
        return self.measured_dimension_px

    @property
    def deviation_mm(self) -> float:
        return self.deviation_px


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def recover_dimensions(
    matched_pairs: list[MatchedPair],
    cad_features: CADFeatureSet,
    transform_result: TransformResult,
    real_gray: np.ndarray | None = None,
) -> list[MeasuredFeature]:
    """
    Produce pixel measurements for every matched feature pair.

    Parameters
    ----------
    matched_pairs    : from feature_matcher.match_features()
    cad_features     : parsed DXF feature set
    transform_result : refined CAD→image transform (used for PCD centre projection)
    real_gray        : original grayscale image (used for active measurement).
                       If None, circle/rect features are skipped.

    All values stored in MeasuredFeature are in IMAGE PIXELS.
    CAD nominals are converted to px using pair.scale_px_per_mm so that
    deviations are meaningful (both sides in the same unit).
    """
    features: list[MeasuredFeature] = []

    # Representative total_scale from matched pairs.
    pair_scales = [p.scale_px_per_mm for p in matched_pairs if p.scale_px_per_mm > 0]
    if pair_scales:
        ts_global = float(np.median(pair_scales))
    else:
        ts_global = transform_result.scale_px_per_mm
        logger.warning(
            f"No valid per-pair scale found; using transform_result scale "
            f"({ts_global:.4f} px/mm)"
        )

    # Hough-detected hole centres (px) — used for spacing / PCD
    detected_centers: dict[str, tuple[float, float]] = {}

    # Track labels measured in the per-pair loop to avoid duplicates
    already_measured_labels: set[str] = set()

    # ── Per-pair measurements ──────────────────────────────────────────────
    for pair in matched_pairs:
        ts = pair.scale_px_per_mm if pair.scale_px_per_mm > 0 else ts_global

        if pair.feature_type == "circle":
            cad_r_mm = float(pair.cad_value_mm)
            cad_r_px = cad_r_mm * ts   # CAD nominal in image pixels

            # Detect whether this circle is the center bore
            is_center_bore = (
                cad_features.part_type == "circular"
                and cad_features.center_bore is not None
                and abs(cad_r_mm * 2.0 - cad_features.center_bore)
                    < cad_features.center_bore * 0.10
            )

            # ── Active measurement: fit a Hough circle in the image ────
            measured_r_px: float
            if is_center_bore:
                # Use Stage-2 image_value_px directly for the center bore.
                # Re-running Hough picks up the larger outer ring instead.
                measured_r_px = float(pair.image_value_px)
                detected_centers[pair.label] = pair.image_pos_px
                logger.debug(
                    f"{pair.label} (center_bore): using Stage-2 r_px={measured_r_px:.1f} "
                    f"(CAD={cad_r_px:.1f}px)"
                )
            elif real_gray is not None:
                detection = _detect_circle_in_roi(
                    real_gray,
                    cx_img_px=pair.image_pos_px[0],
                    cy_img_px=pair.image_pos_px[1],
                    expected_r_px=cad_r_px,
                )
                if detection is not None:
                    cx_det, cy_det, measured_r_px = detection
                    detected_centers[pair.label] = (cx_det, cy_det)
                    logger.debug(
                        f"{pair.label}: Hough measured r={measured_r_px:.1f}px "
                        f"at ({cx_det:.1f},{cy_det:.1f}) (CAD={cad_r_px:.1f}px)"
                    )
                else:
                    logger.warning(
                        f"{pair.label}: Hough circle fit failed in RoI — skipping"
                    )
                    continue
            else:
                logger.warning(
                    f"{pair.label}: no image supplied to recover_dimensions — skipping"
                )
                continue

            if is_center_bore:
                cad_bore_px = cad_features.center_bore * ts
                features.append(MeasuredFeature(
                    feature_type="center_bore",
                    label="center_bore",
                    cad_dimension_px=cad_bore_px,
                    measured_dimension_px=measured_r_px * 2.0,
                    deviation_px=(measured_r_px * 2.0) - cad_bore_px,
                    unit="px (diameter)",
                ))
                already_measured_labels.add(pair.label)
                logger.debug(
                    f"{pair.label} reclassified as center_bore: "
                    f"measured_d={measured_r_px * 2.0:.1f}px (CAD={cad_bore_px:.1f}px)"
                )
            else:
                features.append(MeasuredFeature(
                    feature_type="circle_radius",
                    label=pair.label,
                    cad_dimension_px=cad_r_px,
                    measured_dimension_px=measured_r_px,
                    deviation_px=measured_r_px - cad_r_px,
                    unit="px (radius)",
                ))
                features.append(MeasuredFeature(
                    feature_type="circle_diameter",
                    label=pair.label + "_dia",
                    cad_dimension_px=cad_r_px * 2.0,
                    measured_dimension_px=measured_r_px * 2.0,
                    deviation_px=(measured_r_px - cad_r_px) * 2.0,
                    unit="px (diameter)",
                ))
                already_measured_labels.add(pair.label)

        elif pair.feature_type == "rect":
            cad_wh    = pair.cad_value_mm       # (w_mm, h_mm) from DXF
            cad_w_mm  = float(cad_wh[0])
            cad_h_mm  = float(cad_wh[1])
            cad_w_px  = cad_w_mm * ts           # CAD nominal in px
            cad_h_px  = cad_h_mm * ts

            meas_w_px: float | None = None
            meas_h_px: float | None = None

            if real_gray is not None:
                cx_img = pair.image_pos_px[0]
                cy_img = pair.image_pos_px[1]
                hw_px  = cad_w_px / 2.0
                hh_px  = cad_h_px / 2.0
                search_factor = 2.0
                x1_roi = max(0, int(cx_img - hw_px * search_factor))
                y1_roi = max(0, int(cy_img - hh_px * search_factor))
                x2_roi = min(real_gray.shape[1], int(cx_img + hw_px * search_factor))
                y2_roi = min(real_gray.shape[0], int(cy_img + hh_px * search_factor))
                roi = real_gray[y1_roi:y2_roi, x1_roi:x2_roi]
                if roi.size > 0:
                    blur = cv2.GaussianBlur(roi, (3, 3), 0)

                    def _try_rect_thresh(flags: int):
                        _, thresh_img = cv2.threshold(blur, 0, 255, flags)
                        cnts, _ = cv2.findContours(
                            thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        if not cnts:
                            return None, None
                        lg = max(cnts, key=cv2.contourArea)
                        _, _, wc, hc = cv2.boundingRect(lg)
                        # Sanity check: must be within 40%–160% of expected size
                        wr = wc / cad_w_px if cad_w_px > 0 else 0.0
                        hr = hc / cad_h_px if cad_h_px > 0 else 0.0
                        if 0.4 < wr < 1.6 and 0.4 < hr < 1.6:
                            # Return raw px — no /ts division
                            return float(wc), float(hc)
                        return None, None

                    meas_w_px, meas_h_px = _try_rect_thresh(
                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                    )
                    if meas_w_px is None:
                        meas_w_px, meas_h_px = _try_rect_thresh(
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU
                        )
                    if meas_w_px is not None:
                        logger.debug(
                            f"{pair.label}: contour measured "
                            f"{meas_w_px:.1f}x{meas_h_px:.1f}px "
                            f"(CAD={cad_w_px:.1f}x{cad_h_px:.1f}px)"
                        )

            if meas_w_px is None or meas_h_px is None:
                logger.warning(
                    f"{pair.label}: rect contour measurement failed — skipping"
                )
                continue

            features.append(MeasuredFeature(
                feature_type="rect_width",
                label=pair.label + "_width",
                cad_dimension_px=cad_w_px,
                measured_dimension_px=meas_w_px,
                deviation_px=meas_w_px - cad_w_px,
                unit="px",
            ))
            features.append(MeasuredFeature(
                feature_type="rect_height",
                label=pair.label + "_height",
                cad_dimension_px=cad_h_px,
                measured_dimension_px=meas_h_px,
                deviation_px=meas_h_px - cad_h_px,
                unit="px",
            ))

    pt = cad_features.part_type

    # ── High-level part dimensions ─────────────────────────────────────────
    if pt == "circular":

        if cad_features.outer_diameter is not None:
            od_cad_px = cad_features.outer_diameter * ts_global
            od_meas_px = None
            for p in matched_pairs:
                if p.feature_type == "circle":
                    if p.label in already_measured_labels:
                        continue
                    ts_p   = p.scale_px_per_mm if p.scale_px_per_mm > 0 else ts_global
                    cad_d_px = float(p.cad_value_mm) * 2.0 * ts_p
                    if abs(cad_d_px - od_cad_px) < od_cad_px * 0.15:
                        if real_gray is not None:
                            r_px = _measure_circle_radius_px(
                                real_gray,
                                cx_img_px=p.image_pos_px[0],
                                cy_img_px=p.image_pos_px[1],
                                expected_r_px=float(p.cad_value_mm) * ts_p,
                            )
                            if r_px is not None:
                                od_meas_px = r_px * 2.0
                        break
            if od_meas_px is None:
                logger.warning("outer_diameter: Hough measurement failed, skipping")
            else:
                features.append(MeasuredFeature(
                    feature_type="outer_diameter",
                    label="outer_diameter",
                    cad_dimension_px=od_cad_px,
                    measured_dimension_px=od_meas_px,
                    deviation_px=od_meas_px - od_cad_px,
                    unit="px (diameter)",
                ))

        if cad_features.center_bore is not None:
            cb_cad_px  = cad_features.center_bore * ts_global
            cb_meas_px = None
            for p in matched_pairs:
                if p.feature_type == "circle":
                    if p.label in already_measured_labels:
                        continue
                    ts_p     = p.scale_px_per_mm if p.scale_px_per_mm > 0 else ts_global
                    cad_d_px = float(p.cad_value_mm) * 2.0 * ts_p
                    if abs(cad_d_px - cb_cad_px) < cb_cad_px * 0.10:
                        if real_gray is not None:
                            expected_r_px  = float(p.cad_value_mm) * ts_p
                            tight_min_r    = max(2, int(expected_r_px * 0.75))
                            tight_max_r    = max(tight_min_r + 2, int(expected_r_px * 1.25))
                            h_img, w_img   = real_gray.shape
                            search_r       = int(math.ceil(expected_r_px * _HOUGH_SEARCH_FACTOR))
                            cx_img_px, cy_img_px = p.image_pos_px
                            x1 = max(0, int(cx_img_px) - search_r)
                            y1 = max(0, int(cy_img_px) - search_r)
                            x2 = min(w_img, int(cx_img_px) + search_r)
                            y2 = min(h_img, int(cy_img_px) + search_r)
                            roi = real_gray[y1:y2, x1:x2]
                            if roi.size > 0:
                                blurred = cv2.GaussianBlur(roi, (5, 5), 0)
                                circles = cv2.HoughCircles(
                                    blurred, cv2.HOUGH_GRADIENT,
                                    dp=1.2, minDist=max(4, tight_min_r),
                                    param1=50, param2=18,
                                    minRadius=tight_min_r,
                                    maxRadius=tight_max_r,
                                )
                                if circles is not None:
                                    best_r_px  = None
                                    best_dist  = 1e9
                                    for cx_roi, cy_roi, r_roi in circles[0]:
                                        d = math.hypot(cx_roi + x1 - cx_img_px,
                                                       cy_roi + y1 - cy_img_px)
                                        if d < best_dist:
                                            best_dist = d
                                            best_r_px = float(r_roi)
                                    if best_r_px is not None:
                                        cb_meas_px = best_r_px * 2.0
                        break
            if cb_meas_px is None:
                logger.warning("center_bore: Hough measurement failed, skipping")
            else:
                features.append(MeasuredFeature(
                    feature_type="center_bore",
                    label="center_bore",
                    cad_dimension_px=cb_cad_px,
                    measured_dimension_px=cb_meas_px,
                    deviation_px=cb_meas_px - cb_cad_px,
                    unit="px (diameter)",
                ))

        if cad_features.pcd is not None:
            pcd_cad_px = cad_features.pcd * ts_global

            cx_vals = [c["cx"] for c in cad_features.raw_circles]
            cy_vals = [c["cy"] for c in cad_features.raw_circles]
            if cx_vals:
                part_cx_cad = float(np.median(cx_vals))
                part_cy_cad = float(np.median(cy_vals))
            else:
                part_cx_cad, part_cy_cad = 148.5, 105.0

            cx_img, cy_img = _project_cad_to_image(
                part_cx_cad, part_cy_cad, transform_result.matrix
            )

            CENTER_TOL_MM = 3.0
            bolt_pairs = [
                p for p in matched_pairs
                if p.feature_type == "circle" and
                math.hypot(p.cad_pos[0] - part_cx_cad,
                           p.cad_pos[1] - part_cy_cad) >= CENTER_TOL_MM
            ]

            if len(bolt_pairs) >= 3:
                # Distances are already in px — no /ts needed
                dists_px: list[float] = []
                for p in bolt_pairs:
                    if p.label not in detected_centers:
                        continue
                    bx, by = detected_centers[p.label]
                    dists_px.append(math.hypot(bx - cx_img, by - cy_img))
                if len(dists_px) >= 3:
                    pcd_meas_px = float(np.mean(dists_px)) * 2.0
                    features.append(MeasuredFeature(
                        feature_type="pcd",
                        label="pcd",
                        cad_dimension_px=pcd_cad_px,
                        measured_dimension_px=pcd_meas_px,
                        deviation_px=pcd_meas_px - pcd_cad_px,
                        unit="px (diameter)",
                    ))
                else:
                    logger.warning(
                        f"PCD: only {len(dists_px)} Hough-detected bolt holes "
                        f"(need ≥3), skipping"
                    )
            else:
                logger.warning(
                    f"PCD: only {len(bolt_pairs)} bolt-hole pairs (need ≥3), skipping"
                )

    elif pt == "rectangular":

        if cad_features.overall_width is not None or cad_features.overall_height is not None:
            # _measure_overall_from_mask now returns px directly
            meas_w_px, meas_h_px = _measure_overall_from_mask(real_gray) \
                if real_gray is not None else (None, None)

            if cad_features.overall_width is not None:
                cad_w_px = cad_features.overall_width * ts_global
                if meas_w_px is not None:
                    features.append(MeasuredFeature(
                        feature_type="overall_width",
                        label="overall_width",
                        cad_dimension_px=cad_w_px,
                        measured_dimension_px=meas_w_px,
                        deviation_px=meas_w_px - cad_w_px,
                        unit="px",
                        extra={"note": "measured from image mask bounding box"},
                    ))
                else:
                    logger.warning("overall_width: could not measure from image, skipping")

            if cad_features.overall_height is not None:
                cad_h_px = cad_features.overall_height * ts_global
                if meas_h_px is not None:
                    features.append(MeasuredFeature(
                        feature_type="overall_height",
                        label="overall_height",
                        cad_dimension_px=cad_h_px,
                        measured_dimension_px=meas_h_px,
                        deviation_px=meas_h_px - cad_h_px,
                        unit="px",
                        extra={"note": "measured from image mask bounding box"},
                    ))
                else:
                    logger.warning("overall_height: could not measure from image, skipping")

        # ── Hole spacings ──────────────────────────────────────────────
        hole_pairs        = [p for p in matched_pairs if p.feature_type == "circle"]
        dxf_hole_positions = cad_features.hole_positions

        if len(hole_pairs) >= 2:
            for i in range(len(hole_pairs)):
                for j in range(i + 1, len(hole_pairs)):
                    pi, pj = hole_pairs[i], hole_pairs[j]

                    cxi, cyi = detected_centers.get(pi.label, pi.image_pos_px)
                    cxj, cyj = detected_centers.get(pj.label, pj.image_pos_px)

                    # CAD spacing: DXF positions (mm) → px
                    ts_i = pi.scale_px_per_mm if pi.scale_px_per_mm > 0 else ts_global
                    if i < len(dxf_hole_positions) and j < len(dxf_hole_positions):
                        cad_sp_px = math.hypot(
                            dxf_hole_positions[i][0] - dxf_hole_positions[j][0],
                            dxf_hole_positions[i][1] - dxf_hole_positions[j][1],
                        ) * ts_i
                    else:
                        cad_sp_px = math.hypot(
                            pi.cad_pos[0] - pj.cad_pos[0],
                            pi.cad_pos[1] - pj.cad_pos[1],
                        ) * ts_i

                    # Measured spacing is already in px
                    meas_sp_px = math.hypot(cxi - cxj, cyi - cyj)

                    features.append(MeasuredFeature(
                        feature_type="hole_spacing",
                        label=f"spacing_{pi.label}_to_{pj.label}",
                        cad_dimension_px=cad_sp_px,
                        measured_dimension_px=meas_sp_px,
                        deviation_px=meas_sp_px - cad_sp_px,
                        unit="px",
                    ))

    logger.debug(f"Recovered {len(features)} px measurements")
    return features
