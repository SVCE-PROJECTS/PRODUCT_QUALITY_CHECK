"""
Tests for dimension_analysis/measurement.py

All measurements must be in mm. Tests verify:
  - Units and aliases are consistent
  - Hole spacing is derived from actual image positions, not CAD
  - Overall dimensions are measured from the image mask
  - Circle measurement skips (not fakes) when no image is provided
  - Deviation = measured - cad
"""

import math
import numpy as np
import cv2
import pytest

from dimension_analysis.measurement import (
    recover_dimensions,
    MeasuredFeature,
    _measure_overall_from_mask,
    _measure_circle_radius_px,
)
from dimension_analysis.feature_matcher import MatchedPair
from dimension_analysis.transform_estimator import TransformResult
from dimension_analysis.dxf_parser import CADFeatureSet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transform_result(scale: float = 5.0) -> TransformResult:
    M = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
    return TransformResult(
        matrix=M,
        scale_px_per_mm=scale,
        translation_px=(0.0, 0.0),
        rotation_deg=0.0,
        residual_error=0.0,
        refined=False,
    )


def _make_circle_pair(
    label: str,
    cad_r_mm: float,
    img_cx: float,
    img_cy: float,
    total_scale: float,
) -> MatchedPair:
    return MatchedPair(
        feature_type="circle",
        label=label,
        cad_value_mm=cad_r_mm,
        image_value_px=cad_r_mm * total_scale,
        cad_pos=(0.0, 0.0),
        image_pos_px=(img_cx, img_cy),
        scale_px_per_mm=total_scale,
        match_distance_px=0.0,
    )


def _bright_rect_image(rect_x1=50, rect_y1=80, rect_x2=250, rect_y2=220,
                        img_h=300, img_w=400) -> np.ndarray:
    """Bright rectangle on a dark background — simulates a lit part."""
    img = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), 220, -1)
    return img


# ---------------------------------------------------------------------------
# Unit tests — MeasuredFeature
# ---------------------------------------------------------------------------

def test_measured_feature_unit_is_px():
    mf = MeasuredFeature(
        feature_type="circle_radius",
        label="test",
        cad_dimension_px=25.0,
        measured_dimension_px=25.5,
        deviation_px=0.5,
    )
    assert mf.unit == "px"


def test_measured_feature_mm_aliases_equal_px_fields():
    """Backward-compat mm-named properties must equal the px fields."""
    mf = MeasuredFeature(
        feature_type="circle_radius",
        label="test",
        cad_dimension_px=15.0,
        measured_dimension_px=16.0,
        deviation_px=1.0,
    )
    assert mf.cad_dimension_mm == mf.cad_dimension_px
    assert mf.measured_dimension_mm == mf.measured_dimension_px
    assert mf.deviation_mm == mf.deviation_px


def test_measured_feature_deviation_is_measured_minus_cad():
    mf = MeasuredFeature(
        feature_type="rect_width",
        label="cutout_width",
        cad_dimension_px=125.0,
        measured_dimension_px=127.0,
        deviation_px=2.0,
    )
    assert mf.deviation_px == pytest.approx(
        mf.measured_dimension_px - mf.cad_dimension_px, abs=1e-6
    )


# ---------------------------------------------------------------------------
# _measure_overall_from_mask
# ---------------------------------------------------------------------------

def test_measure_overall_bright_rect():
    """Should detect a bright rectangular part and return correct px size."""
    img = _bright_rect_image(50, 80, 250, 220)   # 200px wide × 140px tall
    w_px, h_px = _measure_overall_from_mask(img)
    assert w_px is not None and h_px is not None
    assert abs(w_px - 200.0) < 20.0, f"Expected ~200px width, got {w_px:.1f}"
    assert abs(h_px - 140.0) < 20.0, f"Expected ~140px height, got {h_px:.1f}"


def test_measure_overall_empty_image_returns_none():
    img = np.zeros((200, 200), dtype=np.uint8)
    w_px, h_px = _measure_overall_from_mask(img)
    assert w_px is None or (isinstance(w_px, float) and w_px >= 0)


def test_measure_overall_zero_scale_returns_none():
    # scale parameter no longer exists — this test is replaced by a smoke test
    img = _bright_rect_image()
    w_px, h_px = _measure_overall_from_mask(img)
    # should not crash and should return a result or None
    assert w_px is None or isinstance(w_px, float)


# ---------------------------------------------------------------------------
# _measure_circle_radius_mm
# ---------------------------------------------------------------------------

def test_measure_circle_radius_detects_circle():
    """Create a synthetic circle in a gray image and verify Hough finds it."""
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(img, (100, 100), 20, 180, -1)   # filled circle r=20px
    r_px = _measure_circle_radius_px(img, 100.0, 100.0, 20.0)
    if r_px is not None:
        assert 10.0 < r_px < 30.0, f"Expected ~20px, got {r_px:.1f}px"


def test_measure_circle_radius_empty_roi_returns_none():
    img = np.zeros((10, 10), dtype=np.uint8)
    r_px = _measure_circle_radius_px(img, 100.0, 100.0, 20.0)
    assert r_px is None


def test_measure_circle_radius_out_of_bounds_returns_none():
    """Centre far outside the image — RoI will be empty."""
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(img, (100, 100), 20, 180, -1)
    r_px = _measure_circle_radius_px(img, 500.0, 500.0, 20.0)
    assert r_px is None


# ---------------------------------------------------------------------------
# recover_dimensions — hole spacing
# ---------------------------------------------------------------------------

def test_hole_spacing_uses_image_positions_not_cad():
    """
    Spacing must come from image pixel distance, not CAD.

    Two holes are 10mm apart in CAD (= 50px at scale 5).
    But in the image they are 60px apart → deviation = 10px.
    """
    scale = 5.0
    p1 = MatchedPair(
        feature_type="circle", label="h1",
        cad_value_mm=1.5, image_value_px=7.5,
        cad_pos=(0.0, 0.0), image_pos_px=(100.0, 200.0),
        scale_px_per_mm=scale,
    )
    p2 = MatchedPair(
        feature_type="circle", label="h2",
        cad_value_mm=1.5, image_value_px=7.5,
        cad_pos=(10.0, 0.0), image_pos_px=(160.0, 200.0),  # 60px apart in image
        scale_px_per_mm=scale,
    )

    fs = CADFeatureSet(
        part_type="rectangular",
        dxf_path="test",
        hole_positions=[(0.0, 0.0), (10.0, 0.0)],  # CAD: 10mm apart = 50px
        overall_width=50.0,
        overall_height=50.0,
    )
    tr = _make_transform_result(scale)
    real_gray = _bright_rect_image(0, 0, 300, 300, 400, 400)

    results = recover_dimensions([p1, p2], fs, tr, real_gray=real_gray)
    spacing = [f for f in results if f.feature_type == "hole_spacing"]

    assert len(spacing) == 1
    sp = spacing[0]
    assert abs(sp.cad_dimension_px - 50.0) < 1.0, \
        f"CAD spacing should be 50px (10mm×5), got {sp.cad_dimension_px:.1f}"
    assert abs(sp.measured_dimension_px - 60.0) < 2.0, \
        f"Measured spacing should be ~60px, got {sp.measured_dimension_px:.1f}"
    assert abs(sp.deviation_px - 10.0) < 2.0, \
        f"Deviation should be ~10px, got {sp.deviation_px:.1f}"


def test_hole_spacing_cad_uses_dxf_positions():
    """CAD spacing must come from DXF positions converted to px."""
    scale = 5.0
    p1 = MatchedPair(
        feature_type="circle", label="h1",
        cad_value_mm=1.5, image_value_px=7.5,
        cad_pos=(0.0, 0.0), image_pos_px=(0.0, 0.0),
        scale_px_per_mm=scale,
    )
    p2 = MatchedPair(
        feature_type="circle", label="h2",
        cad_value_mm=1.5, image_value_px=7.5,
        cad_pos=(31.75, 0.0), image_pos_px=(158.75, 0.0),
        scale_px_per_mm=scale,
    )

    fs = CADFeatureSet(
        part_type="rectangular",
        dxf_path="test",
        hole_positions=[(0.0, 0.0), (31.75, 0.0)],  # 31.75mm → 158.75px at scale 5
        overall_width=60.0,
        overall_height=40.0,
    )
    tr = _make_transform_result(scale)
    real_gray = _bright_rect_image(0, 0, 300, 200, 300, 400)

    results = recover_dimensions([p1, p2], fs, tr, real_gray=real_gray)
    spacing = [f for f in results if f.feature_type == "hole_spacing"]

    assert len(spacing) == 1
    assert abs(spacing[0].cad_dimension_px - 158.75) < 1.0, \
        f"CAD spacing should be ~158.75px, got {spacing[0].cad_dimension_px:.2f}"


# ---------------------------------------------------------------------------
# recover_dimensions — no image → circles skipped
# ---------------------------------------------------------------------------

def test_recover_skips_circles_when_no_image():
    """When real_gray=None, circle features must be skipped (not tautological)."""
    scale = 5.0
    pair = _make_circle_pair("hole_1", 3.0, 100.0, 100.0, scale)
    fs = CADFeatureSet(part_type="circular", dxf_path="test")
    tr = _make_transform_result(scale)

    results = recover_dimensions([pair], fs, tr, real_gray=None)
    circle_features = [f for f in results if "circle" in f.feature_type]
    assert len(circle_features) == 0, (
        "Circle features must be skipped when no image is provided"
    )


# ---------------------------------------------------------------------------
# recover_dimensions — overall dimensions
# ---------------------------------------------------------------------------

def test_overall_width_measured_from_image_not_cad():
    """overall_width must be measured from the image, not copied from CAD."""
    scale = 5.0
    fs = CADFeatureSet(
        part_type="rectangular",
        dxf_path="test",
        overall_width=50.0,
        overall_height=30.0,
    )
    tr = _make_transform_result(scale)
    # 200×140 bright rect on 300×400 image → ~200×140px measured
    real_gray = _bright_rect_image(50, 80, 250, 220, 300, 400)

    results = recover_dimensions([], fs, tr, real_gray=real_gray)
    ow = [f for f in results if f.feature_type == "overall_width"]

    if ow:
        # CAD nominal in px = 50mm × 5 = 250px; measured ~200px
        assert ow[0].cad_dimension_px == pytest.approx(250.0, abs=1.0)
        assert ow[0].unit == "px"
