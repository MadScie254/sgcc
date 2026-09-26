"""The research record: the test-set evaluation, calibration and statistical comparison (read-only)."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from backend.schemas import (
    CalibrationReport, ComparisonRow, ResamplingEffect, ResearchCurve, ResearchDistribution, ResearchEvaluation,
    TrainingSummary,
)
from backend.services import research

router = APIRouter(prefix="/api/research", tags=["research"])


@router.get("/evaluation", response_model=ResearchEvaluation)
def evaluation():
    return research.evaluation()


@router.get("/comparison", response_model=List[ComparisonRow])
def comparison():
    return research.model_comparison()


@router.get("/operating-curve", response_model=ResearchCurve)
def operating_curve():
    return research.operating_curve()


@router.get("/score-distribution", response_model=ResearchDistribution)
def score_distribution():
    return research.score_distribution()


@router.get("/calibration", response_model=CalibrationReport)
def calibration():
    return research.calibration()


@router.get("/significance")
def significance() -> Dict[str, Any]:
    result = research.significance()
    if not result:
        raise HTTPException(status_code=404, detail="No significance record: run python scripts/significance.py")
    return result


@router.get("/resampling", response_model=ResamplingEffect)
def resampling():
    effect = research.resampling_effect()
    if not effect:
        raise HTTPException(status_code=404, detail="No resampling record: run python -m src.train")
    return effect


@router.get("/training", response_model=TrainingSummary)
def training():
    return research.training_summary()
