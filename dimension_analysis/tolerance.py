"""
Tolerance Verification — Stage 7.

PRE-CLAMP PROTOTYPE — PASS/FAIL evaluation is currently DISABLED.
Measurements are collected and passed through without evaluation.

The tolerance evaluation code is preserved below in comments for
future reactivation once the clamp fixture and calibration are added.

To re-enable: uncomment the verify_tolerances body and comment out
the passthrough version.
"""

import logging
from dataclasses import dataclass

from dimension_analysis.measurement import MeasuredFeature

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PRESERVED — tolerances kept for future clamp-phase reactivation
# ---------------------------------------------------------------------------
# DEFAULT_TOLERANCES: dict[str, float] = {
#     "circle_radius":   0.50,
#     "circle_diameter": 1.00,
#     "rect_width":      1.50,
#     "rect_height":     1.50,
#     "hole_spacing":    1.50,
#     "outer_diameter":  2.00,
#     "center_bore":     1.00,
#     "pcd":             2.00,
#     "overall_width":   3.00,
#     "overall_height":  3.00,
#     "_default":        1.00,
# }


@dataclass
class ToleranceResult:
    feature_type: str
    label: str
    cad_dimension_mm: float       # nominal in px (field name kept for compat)
    measured_dimension_mm: float  # measured in px (field name kept for compat)
    deviation_mm: float           # deviation in px (field name kept for compat)
    tolerance_mm: float           # set to 0.0 — evaluation disabled
    status: str                   # always "" — no PASS/FAIL
    unit: str = "px"


def verify_tolerances(
    measured_features: list[MeasuredFeature],
    tolerances: dict[str, float] | None = None,
) -> list[ToleranceResult]:
    """
    PASS/FAIL evaluation is currently DISABLED for the pre-clamp prototype.

    This function passes measured features through as ToleranceResult objects
    without computing any PASS or FAIL status.

    The original evaluation logic is preserved in comments below and can be
    reactivated once the clamp fixture provides stable calibration.
    """
    if not measured_features:
        raise ValueError(
            "Tolerance verification failed: no measured features supplied"
        )

    # ── PASSTHROUGH: no evaluation, no PASS/FAIL ──────────────────────────
    results: list[ToleranceResult] = []
    for mf in measured_features:
        results.append(ToleranceResult(
            feature_type=mf.feature_type,
            label=mf.label,
            cad_dimension_mm=mf.cad_dimension_mm,
            measured_dimension_mm=mf.measured_dimension_mm,
            deviation_mm=mf.deviation_mm,
            tolerance_mm=0.0,   # not evaluated
            status="",          # no PASS/FAIL
            unit=mf.unit,
        ))

    logger.debug(f"Tolerance passthrough: {len(results)} features (no evaluation)")
    return results


# ---------------------------------------------------------------------------
# PRESERVED — original evaluation logic (commented out for future use)
# ---------------------------------------------------------------------------
# def verify_tolerances(
#     measured_features: list[MeasuredFeature],
#     tolerances: dict[str, float] | None = None,
# ) -> list[ToleranceResult]:
#     if not measured_features:
#         raise ValueError("Tolerance verification failed: no measured features supplied")
#
#     tol_map = {**DEFAULT_TOLERANCES}
#     if tolerances:
#         tol_map.update(tolerances)
#
#     results: list[ToleranceResult] = []
#     pass_count = 0
#     fail_count = 0
#
#     for mf in measured_features:
#         tol    = tol_map.get(mf.feature_type, tol_map["_default"])
#         passed = abs(mf.deviation_mm) <= tol
#         status = "PASS" if passed else "FAIL"
#
#         if passed:
#             pass_count += 1
#         else:
#             fail_count += 1
#             logger.warning(
#                 f"FAIL  {mf.label}: "
#                 f"CAD={mf.cad_dimension_mm:.3f}mm  "
#                 f"measured={mf.measured_dimension_mm:.3f}mm  "
#                 f"deviation={mf.deviation_mm:+.3f}mm  "
#                 f"tolerance=±{tol:.2f}mm"
#             )
#
#         results.append(ToleranceResult(
#             feature_type=mf.feature_type,
#             label=mf.label,
#             cad_dimension_mm=mf.cad_dimension_mm,
#             measured_dimension_mm=mf.measured_dimension_mm,
#             deviation_mm=mf.deviation_mm,
#             tolerance_mm=tol,
#             status=status,
#             unit=mf.unit,
#         ))
#
#     logger.debug(f"Tolerance check: {pass_count} PASS, {fail_count} FAIL")
#     return results
