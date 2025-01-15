# tests/test_fitters/test_drt.py
import pytest
import numpy as np
from impedance_agent.core.models import ImpedanceData, DRTResult
from impedance_agent.fitters.drt import DRTFitter


@pytest.fixture
def single_tau_data():
    """Generate synthetic data with single time constant"""
    freq = np.logspace(-2, 5, 50)
    tau = 1e-3  # 1ms time constant
    R = 100.0  # Large resistance for better signal
    w = 2 * np.pi * freq
    Z = R / (1 + 1j * w * tau)

    return ImpedanceData(frequency=freq, real=Z.real, imaginary=Z.imag)


@pytest.fixture
def drt_fitter(single_tau_data):
    """Create DRT fitter instance with default parameters"""
    freq = single_tau_data.frequency
    return DRTFitter(
        zexp_re=single_tau_data.real,
        zexp_im=single_tau_data.imaginary,
        omg=2 * np.pi * freq,
        lam_t0=1e-6,  # Smaller regularization for sharper peaks
        lam_pg0=1e-5,
        lower_bounds=np.array([1e-10, 1e-10]),
        upper_bounds=np.array([1e3, 1e3]),
        mode="real",
    )


def test_drt_initialization(drt_fitter):
    """Test DRT fitter initialization"""
    assert drt_fitter is not None
    assert drt_fitter.mode == "real"
    assert drt_fitter.lam_t0 == 1e-6
    assert drt_fitter.lam_pg0 == 1e-5
    assert drt_fitter.niter == 80
    assert hasattr(drt_fitter, "tau")
    assert hasattr(drt_fitter, "ln_tau")
    assert hasattr(drt_fitter, "d_ln_tau")


def test_drt_mesh_creation(drt_fitter):
    """Test mesh creation"""
    assert drt_fitter.d_ln_tau.shape == (50,)
    assert drt_fitter.d_tau.shape == (50,)
    # Test endpoint handling
    assert np.isfinite(drt_fitter.d_ln_tau[0])
    assert np.isfinite(drt_fitter.d_ln_tau[-1])


def test_drt_matrix_creation(drt_fitter):
    """Test creation of DRT system matrices"""
    assert drt_fitter.a_matrix.shape == (50, 50)
    assert drt_fitter.a_matrix_t.shape == (50, 50)
    assert drt_fitter.a_mat_t_a.shape == (50, 50)
    assert drt_fitter.b_rhs.shape == (50,)
    # Test symmetry of A^T A
    assert np.allclose(drt_fitter.a_mat_t_a, drt_fitter.a_mat_t_a.T)


def test_drt_fitting(drt_fitter):
    """Test DRT fitting with single time constant"""
    result = drt_fitter.fit()

    assert isinstance(result, DRTResult)
    assert len(result.peak_frequencies) > 0

    # Check basic result properties
    assert result.tau.shape == (50,)
    assert result.gamma.shape == (50,)
    assert isinstance(result.regularization_param, float)

    # Check fitted impedance
    assert result.Z_fit is not None
    assert result.Z_fit.shape == (50,)

    # Check residuals are finite
    assert np.all(np.isfinite(result.residuals_real))
    assert np.all(np.isfinite(result.residuals_imag))


def test_find_lambda(drt_fitter):
    """Test lambda parameter search"""
    resid, solnorm, lam_t_arr, lam_pg_arr = drt_fitter.find_lambda()

    assert resid.shape == (25,)
    assert solnorm.shape == (25,)
    assert lam_t_arr.shape == (25,)
    assert lam_pg_arr.shape == (25,)
    assert np.all(np.isfinite(resid))
    assert np.all(np.isfinite(solnorm))


def test_compute_normalized_residuals(drt_fitter):
    """Test calculation of normalized residuals"""
    # Create dummy fitted impedance
    Z_fit = np.ones_like(drt_fitter.zexp_re) + 1j * np.ones_like(drt_fitter.zexp_im)

    residuals_real, residuals_imag = drt_fitter.compute_normalized_residuals(Z_fit)

    assert residuals_real.shape == (50,)
    assert residuals_imag.shape == (50,)
    assert np.all(np.isfinite(residuals_real))
    assert np.all(np.isfinite(residuals_imag))


def test_z_model_imre(drt_fitter):
    """Test impedance model calculation"""
    # Use initial guess for testing
    g_vector = drt_fitter.gfun_init
    Z_model = drt_fitter.z_model_imre(g_vector)

    assert Z_model.shape == (50,)
    assert np.all(np.isfinite(Z_model))
