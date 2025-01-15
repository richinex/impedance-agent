# tests/test_fitters/test_ecm.py
import pytest
import numpy as np
import jax.numpy as jnp
from impedance_agent.core.models import ImpedanceData, FitResult
from impedance_agent.fitters.ecm import ECMFitter


@pytest.fixture
def randles_data():
    """Generate synthetic Randles circuit data"""
    freq = np.logspace(-2, 5, 50)
    Rs = 1.0
    Rct = 2.0
    Cdl = 1e-3
    w = 2 * np.pi * freq
    Z = Rs + Rct / (1 + 1j * w * Cdl * Rct)
    return ImpedanceData(frequency=freq, real=Z.real, imaginary=Z.imag)


@pytest.fixture
def randles_model():
    """Define Randles circuit model function"""

    def model(p, f):
        w = 2 * jnp.pi * f
        Rs, Rct, Cdl = p
        Z = Rs + Rct / (1 + 1j * w * Cdl * Rct)
        return jnp.concatenate([Z.real, Z.imag])

    return model


@pytest.fixture
def ecm_fitter(randles_data, randles_model):
    """Create ECM fitter instance"""
    param_info = [
        {"name": "Rs", "units": "Ω"},
        {"name": "Rct", "units": "Ω"},
        {"name": "Cdl", "units": "F"},
    ]

    return ECMFitter(
        model_func=randles_model,
        p0=np.array([0.5, 1.0, 1e-4]),
        freq=randles_data.frequency,
        impedance_data=randles_data,
        lb=np.array([0.0, 0.0, 0.0]),
        ub=np.array([10.0, 10.0, 1.0]),
        param_info=param_info,
        weighting="modulus",
    )


def test_ecm_initialization(ecm_fitter):
    """Test ECM fitter initialization"""
    assert ecm_fitter is not None
    assert ecm_fitter.num_params == 3
    assert ecm_fitter.num_freq == 50
    assert ecm_fitter.dof == 97  # 2*50 - 3


def test_ecm_fitting(ecm_fitter):
    """Test ECM fitting with Randles circuit"""
    result = ecm_fitter.fit()

    assert isinstance(result, FitResult)
    assert len(result.parameters) == 3

    # Check parameter values
    Rs, Rct, Cdl = result.parameters
    assert 0.9 < Rs < 1.1
    assert 1.9 < Rct < 2.1
    assert 0.9e-3 < Cdl < 1.1e-3

    # Check fit quality
    assert result.wrms < 1e-4
    assert result.fit_quality is not None
    assert result.fit_quality.vector_difference < 0.05
    assert result.fit_quality.path_deviation < 0.05


def test_residuals_calculation(ecm_fitter):
    """Test calculation of normalized residuals"""
    Z_fit = 1.0 + 2.0 / (1 + 1j * 2 * np.pi * ecm_fitter.freq * 1e-3 * 2.0)
    residuals_real, residuals_imag = ecm_fitter.compute_normalized_residuals(Z_fit)

    assert residuals_real.shape == (50,)
    assert residuals_imag.shape == (50,)
    assert np.all(np.abs(residuals_real) < 0.01)
    assert np.all(np.abs(residuals_imag) < 0.01)


def test_invalid_weighting():
    """Test ECM fitter with invalid weighting"""
    with pytest.raises(AssertionError):
        ECMFitter(
            model_func=lambda p, f: p[0],
            p0=np.array([1.0]),
            freq=np.array([1.0]),
            impedance_data=ImpedanceData(
                frequency=np.array([1.0]),
                real=np.array([1.0]),
                imaginary=np.array([0.0]),
            ),
            lb=np.array([0.0]),
            ub=np.array([10.0]),
            param_info=[{"name": "R"}],
            weighting="invalid",
        )
