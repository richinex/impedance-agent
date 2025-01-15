# # src/core/models.py


from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
import numpy as np
from datetime import datetime

class ImpedanceData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    frequency: np.ndarray
    real: np.ndarray
    imaginary: np.ndarray
    measurement_id: Optional[str] = None
    timestamp: Optional[datetime] = None

class FitQualityMetrics(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Vector difference metrics
    vector_difference: float  # Overall magnitude/direction mismatch
    vector_quality: str      # "excellent", "acceptable", "poor"

    # Path following metrics
    path_deviation: float    # How well fit follows data trajectory
    path_quality: str       # "excellent", "acceptable", "poor"

    # Overall assessment
    overall_quality: str    # "excellent", "acceptable", "poor"

class FitResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    parameters: List[float]
    errors: List[float]
    param_info: List[Dict]
    correlation_matrix: Optional[np.ndarray] = None  # Add this field
    chi_square: float
    aic: float
    wrms: float
    dof: int
    measurement_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    Z_fit: Optional[np.ndarray] = None
    fit_quality: Optional[FitQualityMetrics] = None

class DRTResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tau: np.ndarray
    gamma: np.ndarray
    peak_frequencies: List[float]
    peak_polarizations: List[float]
    regularization_param: float
    residual: float
    Z_fit: Optional[np.ndarray] = None
    residuals_real: Optional[np.ndarray] = None
    residuals_imag: Optional[np.ndarray] = None
    fit_quality: Optional[FitQualityMetrics] = None

class LinKKResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    M: int
    mu: float
    Z_fit: np.ndarray
    residuals_real: np.ndarray
    residuals_imag: np.ndarray
    max_residual: float
    mean_residual: float
    fit_quality: Optional[FitQualityMetrics] = None

class AnalysisResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    ecm_fit: Optional[FitResult] = None
    drt_fit: Optional[DRTResult] = None
    linkk_fit: Optional[LinKKResult] = None
    summary: str = ""
    recommendations: List[str] = Field(default_factory=list)
    overall_assessment: Dict[str, Any] = Field(
        default_factory=dict,
        description="Overall analysis including cross-method fit quality comparison"
    )