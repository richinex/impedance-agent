# src/core/models.py
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
import numpy as np
from datetime import datetime

class ImpedanceData(BaseModel):
    """Container for electrochemical impedance spectroscopy data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frequency: np.ndarray = Field(description="Array of measurement frequencies in Hz")
    real: np.ndarray = Field(description="Real component of impedance in ohms")
    imaginary: np.ndarray = Field(description="Imaginary component of impedance in ohms")
    measurement_id: Optional[str] = Field(None, description="Unique identifier for the measurement")
    timestamp: Optional[datetime] = Field(None, description="When the measurement was taken")

class FitQualityMetrics(BaseModel):
    """Quality metrics for assessing impedance fits.

    References
    ----------
    [1] Boukamp, B.A. "A Linear Kronig-Kramers Transform Test for Immittance Data Validation."
        J. Electrochem. Soc. 142 (1995)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_difference: float = Field(description="Quantified vector difference metric")
    vector_quality: str = Field(description='Assessment of vector matching quality ("excellent", "acceptable", "poor")')
    path_deviation: float = Field(description="Quantified path following metric")
    path_quality: str = Field(description='Assessment of path following quality ("excellent", "acceptable", "poor")')
    overall_quality: str = Field(description="Combined quality assessment")

class FitResult(BaseModel):
    """Results from equivalent circuit model fitting.

    Notes
    -----
    The fitting process uses weighted complex nonlinear least squares with:

    * Parameter estimation minimizing: χ² = Σᵢ wᵢ|Z_{exp,i} - Z_{fit,i}|²
    * Akaike Information Criterion: AIC = -2ln(L) + 2k where L is likelihood and k is number of parameters
    * Parameter uncertainties via QR decomposition of weighted Jacobian
    * Correlation matrix from Hessian of objective function

    References
    ----------
    [1] Sadkowski, A. "CNLS fits and Kramers-Kronig validation of resonant EIS data."
        Journal of Electroanalytical Chemistry (2004)
    [2] Ingdal, M., Johnsen, R., & Harrington, D. A. "The Akaike information criterion
        in weighted regression of immittance data." Electrochimica Acta (2019)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    parameters: List[float] = Field(description="Optimized model parameters")
    errors: List[float] = Field(description="Parameter uncertainties (standard errors)")
    param_info: List[Dict] = Field(description="Parameter names and bounds")
    correlation_matrix: Optional[np.ndarray] = Field(None, description="Parameter correlation coefficients")
    chi_square: float = Field(description="Chi-square statistic")
    aic: float = Field(description="Akaike Information Criterion")
    wrms: float = Field(description="Weighted root mean square error")
    dof: int = Field(description="Degrees of freedom")
    measurement_id: Optional[str] = Field(None, description="Unique identifier for the measurement")
    timestamp: Optional[datetime] = Field(None, description="When the fit was performed")
    Z_fit: Optional[np.ndarray] = Field(None, description="Fitted impedance values")
    fit_quality: Optional[FitQualityMetrics] = Field(None, description="Detailed quality metrics")

class DRTResult(BaseModel):
    """Results from Distribution of Relaxation Times (DRT) analysis.

    Notes
    -----
    The DRT analysis solves the integral equation:
    Z(ω) - R_∞ = R_pol ∫₀^∞ γ(τ)/(1 + iωτ)dτ

    Uses Tikhonov Regularization + Projected Gradient (TRPG) method.

    References
    ----------
    [1] Kulikovsky, A. "PEM fuel cell distribution of relaxation times: a method for the
        calculation and behavior of an oxygen transport peak." (2021)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tau: np.ndarray = Field(description="Time constant array")
    gamma: np.ndarray = Field(description="DRT function values")
    peak_frequencies: List[float] = Field(description="Characteristic frequencies of identified processes")
    peak_polarizations: List[float] = Field(description="Polarization resistance of each process")
    regularization_param: float = Field(description="Optimal regularization parameter")
    residual: float = Field(description="Fitting residual")
    Z_fit: Optional[np.ndarray] = Field(None, description="Reconstructed impedance from DRT")
    residuals_real: Optional[np.ndarray] = Field(None, description="Real part fitting residuals")
    residuals_imag: Optional[np.ndarray] = Field(None, description="Imaginary part fitting residuals")
    fit_quality: Optional[FitQualityMetrics] = Field(None, description="Quality assessment metrics")

class LinKKResult(BaseModel):
    """Results from Lin-KK (Linear Kramers-Kronig) validation analysis.

    References
    ----------
    [1] Boukamp, B.A. "A Linear Kronig‐Kramers Transform Test for Immittance Data Validation."
        J. Electrochem. Soc. 142 (1995)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    M: int = Field(description="Number of basis functions used")
    mu: float = Field(description="Spacing parameter for basis functions")
    Z_fit: np.ndarray = Field(description="K-K consistent impedance")
    residuals_real: np.ndarray = Field(description="Real part residuals")
    residuals_imag: np.ndarray = Field(description="Imaginary part residuals")
    max_residual: float = Field(description="Maximum absolute residual")
    mean_residual: float = Field(description="Mean absolute residual")
    fit_quality: Optional[FitQualityMetrics] = Field(None, description="Quality assessment metrics")

class AnalysisResult(BaseModel):
    """Comprehensive impedance analysis results combining multiple methods.

    References
    ----------
    [1] Barsoukov, E., Macdonald, J.R. "Impedance Spectroscopy: Theory, Experiment, and Applications."
        Wiley (2018)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ecm_fit: Optional[FitResult] = Field(None, description="Equivalent circuit fitting results")
    drt_fit: Optional[DRTResult] = Field(None, description="DRT analysis results")
    linkk_fit: Optional[LinKKResult] = Field(None, description="Data validation results")
    summary: str = Field("", description="Key findings and conclusions")
    recommendations: List[str] = Field(default_factory=list, description="Suggested next steps and improvements")
    overall_assessment: Dict[str, Any] = Field(
        default_factory=dict,
        description="Overall analysis including cross-method fit quality comparison"
    )