# tests/test_core/test_loaders.py
import pytest
import numpy as np
import pandas as pd
from impedance_agent.core.loaders import ImpedanceLoader
from impedance_agent.core.exceptions import DataLoadError
from impedance_agent.core.models import ImpedanceData


@pytest.fixture
def sample_data_variations():
    """Generate sample data in different formats"""
    return [
        # Basic format
        "freq,zreal,zimag\n1000,1.0,-0.5\n100,1.5,-1.0\n10,2.0,-1.5",
        # No header
        "1000,1.0,-0.5\n100,1.5,-1.0\n10,2.0,-1.5",
        # Tab separated
        "freq\tzreal\tzimag\n1000\t1.0\t-0.5\n100\t1.5\t-1.0\n10\t2.0\t-1.5",
        # Space separated
        "freq zreal zimag\n1000 1.0 -0.5\n100 1.5 -1.0\n10 2.0 -1.5",
    ]


@pytest.fixture
def sample_txt_files(tmp_path, sample_data_variations):
    """Create sample txt data files"""
    files = []
    for i, data in enumerate(sample_data_variations):
        file_path = tmp_path / f"test_data_{i}.txt"
        file_path.write_text(data)
        files.append(file_path)
    return files


@pytest.fixture
def sample_csv_file(tmp_path):
    """Create sample CSV data file"""
    file_path = tmp_path / "test_data.csv"
    data = pd.DataFrame(
        {"freq": [1000, 100, 10], "zreal": [1.0, 1.5, 2.0], "zimag": [-0.5, -1.0, -1.5]}
    )
    data.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def sample_excel_file(tmp_path):
    """Create sample Excel data file"""
    file_path = tmp_path / "test_data.xlsx"
    data = pd.DataFrame(
        {"freq": [1000, 100, 10], "zreal": [1.0, 1.5, 2.0], "zimag": [-0.5, -1.0, -1.5]}
    )
    data.to_excel(file_path, index=False)
    return file_path


def test_load_txt_files(sample_txt_files):
    """Test loading from txt files with different formats"""
    for file_path in sample_txt_files:
        data = ImpedanceLoader.load(file_path)

        assert isinstance(data, ImpedanceData)
        assert len(data.frequency) == 3
        assert np.allclose(data.frequency, [1000, 100, 10])
        assert np.allclose(data.real, [1.0, 1.5, 2.0])
        assert np.allclose(data.imaginary, [-0.5, -1.0, -1.5])
        assert np.all(np.diff(data.frequency) < 0)  # Check descending order


def test_load_csv_file(sample_csv_file):
    """Test loading from CSV file"""
    data = ImpedanceLoader.load(sample_csv_file)

    assert isinstance(data, ImpedanceData)
    assert len(data.frequency) == 3
    assert np.allclose(data.frequency, [1000, 100, 10])
    assert np.allclose(data.real, [1.0, 1.5, 2.0])
    assert np.allclose(data.imaginary, [-0.5, -1.0, -1.5])


def test_load_excel_file(sample_excel_file):
    """Test loading from Excel file"""
    data = ImpedanceLoader.load(sample_excel_file)

    assert isinstance(data, ImpedanceData)
    assert len(data.frequency) == 3
    assert np.allclose(data.frequency, [1000, 100, 10])
    assert np.allclose(data.real, [1.0, 1.5, 2.0])
    assert np.allclose(data.imaginary, [-0.5, -1.0, -1.5])


def test_invalid_file():
    """Test loading nonexistent file"""
    with pytest.raises(DataLoadError, match="File not found"):
        ImpedanceLoader.load("nonexistent.txt")


def test_unsupported_format(tmp_path):
    """Test loading unsupported file format"""
    invalid_file = tmp_path / "data.invalid"
    invalid_file.touch()

    with pytest.raises(DataLoadError, match="Unsupported file format"):
        ImpedanceLoader.load(invalid_file)


def test_invalid_data(tmp_path):
    """Test loading file with invalid data"""
    file_path = tmp_path / "invalid_data.txt"
    file_path.write_text("not,valid,data\na,b,c\n")

    with pytest.raises(DataLoadError):
        ImpedanceLoader.load(file_path)


def test_frequency_ordering(tmp_path):
    """Test automatic frequency reordering"""
    file_path = tmp_path / "unordered.csv"
    data = pd.DataFrame(
        {
            "freq": [10, 1000, 100],  # Unordered frequencies
            "zreal": [2.0, 1.0, 1.5],
            "zimag": [-1.5, -0.5, -1.0],
        }
    )
    data.to_csv(file_path, index=False)

    result = ImpedanceLoader.load(file_path)
    assert np.all(np.diff(result.frequency) < 0)  # Check descending order
    assert np.allclose(result.frequency, [1000, 100, 10])
    assert np.allclose(result.real, [1.0, 1.5, 2.0])
    assert np.allclose(result.imaginary, [-0.5, -1.0, -1.5])
