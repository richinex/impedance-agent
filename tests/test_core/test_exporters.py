# tests/test_core/test_exporters.py
import pytest
import json
import pandas as pd
import numpy as np
from pathlib import Path
from impedance_agent.core.exporters import ResultExporter
from impedance_agent.core.models import (
    AnalysisResult,
    FitResult,
    DRTResult,
    FitQualityMetrics,
)
from impedance_agent.core.exceptions import ExportError
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def sample_fit_quality():
    """Generate sample fit quality metrics"""
    return FitQualityMetrics(
        vector_difference=0.02,
        vector_quality="excellent",
        path_deviation=0.03,
        path_quality="excellent",
        overall_quality="excellent",
    )


@pytest.fixture
def sample_ecm_fit(sample_fit_quality):
    """Generate sample ECM fit results"""
    return FitResult(
        parameters=[1.0, 2.0, 3.0],
        errors=[0.1, 0.2, 0.3],
        param_info=[{"name": "Rs"}, {"name": "Rct"}, {"name": "Cdl"}],
        wrms=0.001,
        chi_square=0.05,
        aic=100.0,
        dof=10,
        Z_fit=np.array([1 + 1j, 2 + 2j]),
        fit_quality=sample_fit_quality,
    )


@pytest.fixture
def sample_drt_fit(sample_fit_quality):
    """Generate sample DRT fit results"""
    return DRTResult(
        tau=np.array([1.0, 10.0, 100.0]),
        gamma=np.array([0.1, 0.2, 0.3]),
        peak_frequencies=[10.0, 100.0],
        peak_polarizations=[0.2, 0.3],
        regularization_param=1e-3,
        residual=0.001,
        Z_fit=np.array([1 + 1j, 2 + 2j]),
        residuals_real=np.array([0.01, 0.02]),
        residuals_imag=np.array([0.01, 0.02]),
        fit_quality=sample_fit_quality,
    )


@pytest.fixture
def sample_result(sample_ecm_fit, sample_drt_fit):
    """Generate sample analysis result"""
    return AnalysisResult(
        summary="Test analysis summary",
        recommendations=["Recommendation 1", "Recommendation 2"],
        ecm_fit=sample_ecm_fit,
        drt_fit=sample_drt_fit,
        overall_assessment={"quality": "excellent", "confidence": "high"},
    )


@pytest.mark.asyncio
async def test_export_json(tmp_path, sample_result):
    """Test JSON export"""
    file_path = tmp_path / "test.json"
    await ResultExporter.export_async(sample_result, file_path, "json")

    assert file_path.exists()
    with open(file_path) as f:
        data = json.load(f)

    assert data["summary"] == "Test analysis summary"
    assert len(data["recommendations"]) == 2
    assert len(data["ecm_fit"]["parameters"]) == 3
    assert len(data["drt_fit"]["tau"]) == 3
    assert data["ecm_fit"]["fit_quality"]["overall_quality"] == "excellent"
    assert data["overall_assessment"]["quality"] == "excellent"


@pytest.mark.asyncio
async def test_export_csv(tmp_path, sample_result):
    """Test CSV export"""
    file_path = tmp_path / "test.csv"
    await ResultExporter.export_async(sample_result, file_path, "csv")

    # Check ECM CSV
    ecm_path = tmp_path / "test_ecm.csv"
    assert ecm_path.exists()
    ecm_data = pd.read_csv(ecm_path)
    assert len(ecm_data) == 3
    assert all(col in ecm_data.columns for col in ["Parameter", "Value", "Error"])
    assert list(ecm_data["Parameter"]) == ["Rs", "Rct", "Cdl"]

    # Check DRT CSV
    drt_path = tmp_path / "test_drt.csv"
    assert drt_path.exists()
    drt_data = pd.read_csv(drt_path)
    assert len(drt_data) == 3
    assert all(col in drt_data.columns for col in ["tau", "gamma"])

    # Check summary file exists and contains content
    summary_path = tmp_path / "test_summary.txt"
    assert summary_path.exists()
    with open(summary_path) as f:
        summary = f.read()
    assert summary == sample_result.summary


@pytest.mark.asyncio
async def test_export_markdown(tmp_path, sample_result):
    """Test Markdown export"""
    file_path = tmp_path / "test.md"
    await ResultExporter.export_async(sample_result, file_path, "md")

    assert file_path.exists()
    content = file_path.read_text()

    # Check main sections
    assert "# Impedance Analysis Summary" in content
    assert "## Recommendations" in content
    assert "## ECM Fit Metrics" in content
    assert "### Fitted Parameters" in content

    # Check content
    assert "* WRMS: " in content
    assert "* χ²: " in content
    assert "* AIC: " in content
    assert "| Parameter | Value | Error |" in content
    assert "| Rs |" in content
    assert "| Rct |" in content
    assert "| Cdl |" in content


@pytest.mark.asyncio
async def test_empty_result(tmp_path):
    """Test export with minimal result object"""
    result = AnalysisResult(
        summary="Minimal test", recommendations=[], overall_assessment={}
    )

    # Test each format with minimal data
    for fmt in ["json", "md"]:  # These formats always work
        file_path = tmp_path / f"test.{fmt}"
        await ResultExporter.export_async(result, file_path, fmt)
        assert file_path.exists()

    # CSV should at least create a summary file
    await ResultExporter.export_async(result, tmp_path / "test.csv", "csv")
    assert (tmp_path / "test_summary.txt").exists()

    # Excel should create a file with summary sheet
    await ResultExporter.export_async(result, tmp_path / "test.xlsx", "excel")
    assert (tmp_path / "test.xlsx").exists()


@pytest.mark.asyncio
async def test_invalid_format(tmp_path, sample_result):
    """Test invalid export format"""
    with pytest.raises(ExportError, match="Unsupported export format"):
        await ResultExporter.export_async(
            sample_result, tmp_path / "test.txt", "invalid"
        )


@pytest.mark.asyncio
async def test_io_error_handling(tmp_path, sample_result, monkeypatch):
    """Test handling of IO errors during export"""

    def mock_open(*args, **kwargs):
        raise IOError("Simulated IO error")

    monkeypatch.setattr("aiofiles.open", mock_open)

    with pytest.raises(
        ExportError, match="JSON export failed"
    ):  # Changed to match actual error
        await ResultExporter.export_async(sample_result, tmp_path / "test.json", "json")


@pytest.mark.asyncio
async def test_complex_drt_data(tmp_path, sample_fit_quality):
    """Test export with complex DRT data"""
    drt_result = DRTResult(
        tau=np.array([1.0, 10.0]),
        gamma=np.array([1 + 2j, 3 + 4j]),
        peak_frequencies=[10.0],
        peak_polarizations=[0.2],
        regularization_param=1e-3,
        residual=0.001,
        Z_fit=np.array([1 + 1j, 2 + 2j]),
        residuals_real=np.array([0.01, 0.02]),
        residuals_imag=np.array([0.01, 0.02]),
        fit_quality=sample_fit_quality,
    )

    result = AnalysisResult(
        summary="Complex DRT test",
        recommendations=[],
        drt_fit=drt_result,
        overall_assessment={},
    )

    # Test JSON export handles complex numbers
    json_path = tmp_path / "test.json"
    await ResultExporter.export_async(result, json_path, "json")
    with open(json_path) as f:
        data = json.load(f)
    assert "real" in data["drt_fit"]["gamma"][0]
    assert "imag" in data["drt_fit"]["gamma"][0]


@pytest.mark.asyncio
async def test_permission_error(tmp_path, sample_result):
    """Test handling of permission errors"""
    if hasattr(Path, "chmod"):  # Skip on Windows
        file_path = tmp_path / "test.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.parent.chmod(0o444)  # Make directory read-only

        with pytest.raises(
            ExportError, match="JSON export failed.*Permission denied"
        ):  # Changed to match actual error
            await ResultExporter.export_async(sample_result, file_path, "json")

        # Cleanup
        file_path.parent.chmod(0o755)


@pytest.mark.asyncio
async def test_excel_export_complete(tmp_path, sample_result):
    """Test complete Excel export with all sheets"""
    file_path = tmp_path / "test.xlsx"
    await ResultExporter.export_async(sample_result, file_path, "excel")

    xlsx = pd.read_excel(file_path, sheet_name=None)
    assert set(xlsx.keys()) == {"ECM_Fit", "DRT_Results", "Summary"}

    # Verify each sheet's content
    assert len(xlsx["ECM_Fit"]) == 3
    assert len(xlsx["DRT_Results"]) == 3
    assert len(xlsx["Summary"]) == 1
