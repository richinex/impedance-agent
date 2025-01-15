# tests/test_fitters/test_linkk.py
import pytest
import numpy as np
from impedance_agent.core.models import ImpedanceData, LinKKResult
from impedance_agent.fitters.linkk import LinKKFitter

@pytest.fixture
def simple_rc_data():
    """Generate synthetic RC circuit data that should be KK-compliant"""
    freq = np.logspace(-2, 5, 50)
    R = 100.0  # Resistance in ohms
    C = 1e-6   # Capacitance in farads
    w = 2 * np.pi * freq
    Z = R / (1 + 1j * w * R * C)

    return ImpedanceData(
        frequency=freq,
        real=Z.real,
        imaginary=Z.imag
    )

@pytest.fixture
def linkk_fitter(simple_rc_data):
    """Create LinKK fitter instance"""
    return LinKKFitter(data=simple_rc_data)

def test_linkk_initialization(simple_rc_data):
    """Test LinKK fitter initialization"""
    fitter = LinKKFitter(data=simple_rc_data)

    assert fitter is not None
    assert np.array_equal(fitter.freq, simple_rc_data.frequency)
    assert np.array_equal(fitter.Z.real, simple_rc_data.real)
    assert np.array_equal(fitter.Z.imag, simple_rc_data.imaginary)

def test_linkk_fitting(linkk_fitter):
    """Test LinKK fitting with RC circuit data"""
    result = linkk_fitter.fit(c=0.85, max_M=50)

    assert isinstance(result, LinKKResult)
    assert result.M > 0  # Should find some RC elements
    assert result.M <= 50  # Shouldn't exceed max_M

    # Check residuals - relaxed thresholds for numerical stability
    assert result.max_residual < 0.5
    assert result.mean_residual < 0.2

    # Check fitted impedance shape
    assert result.Z_fit.shape == linkk_fitter.Z.shape

    # Residuals arrays should match data length
    assert len(result.residuals_real) == len(linkk_fitter.freq)
    assert len(result.residuals_imag) == len(linkk_fitter.freq)

def test_linkk_parameters(linkk_fitter):
    """Test LinKK fitting with different parameters"""
    # Test with different cutoff ratios
    result1 = linkk_fitter.fit(c=0.80, max_M=50)
    result2 = linkk_fitter.fit(c=0.90, max_M=50)

    # Check each result is valid
    assert result1 is not None
    assert result2 is not None
    assert result1.M > 0
    assert result2.M > 0

    # Test with limited max_M
    result3 = linkk_fitter.fit(max_M=10)
    assert result3 is not None
    assert result3.M <= 10

def test_linkk_with_noisy_data(simple_rc_data):
    """Test LinKK fitting with noisy data"""
    # Generate noisy RC data
    np.random.seed(42)  # For reproducibility
    noise_level = 0.02  # 2% noise
    Z_noisy = (simple_rc_data.real + 1j * simple_rc_data.imaginary) * \
              (1 + noise_level * (np.random.randn(len(simple_rc_data.frequency)) + \
               1j * np.random.randn(len(simple_rc_data.frequency))))

    noisy_data = ImpedanceData(
        frequency=simple_rc_data.frequency,
        real=Z_noisy.real,
        imaginary=Z_noisy.imag
    )

    fitter = LinKKFitter(data=noisy_data)
    result = fitter.fit()

    assert result is not None
    assert result.mean_residual < 0.5  # Relaxed threshold for noisy data

def test_linkk_error_handling():
    """Test LinKK error handling"""
    # Test with invalid frequency data (single point)
    invalid_data = ImpedanceData(
        frequency=np.array([1.0]),  # Single frequency point
        real=np.array([1.0]),
        imaginary=np.array([0.0])
    )

    fitter = LinKKFitter(data=invalid_data)
    result = fitter.fit()
    assert result is None  # Should return None for invalid data

def test_linkk_consistency(linkk_fitter):
    """Test consistency of LinKK results"""
    # Run multiple fits with same parameters
    results = []
    for _ in range(3):
        result = linkk_fitter.fit(c=0.85, max_M=50)
        assert result is not None
        results.append(result)

    # Check number of elements is consistent
    Ms = [r.M for r in results]
    assert max(Ms) - min(Ms) <= 1  # Allow for small variations