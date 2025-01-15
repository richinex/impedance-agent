# src/core/loaders.py
from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
from .models import ImpedanceData
from .exceptions import DataLoadError


class ImpedanceLoader:
    """Handles loading impedance data from various file formats"""

    @staticmethod
    def load(file_path: Union[str, Path]) -> ImpedanceData:
        """Load impedance data from file based on extension"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise DataLoadError(f"File not found: {file_path}")

        try:
            if file_path.suffix == ".txt":
                return ImpedanceLoader._load_txt(file_path)
            elif file_path.suffix == ".csv":
                return ImpedanceLoader._load_csv(file_path)
            elif file_path.suffix == ".xlsx":
                return ImpedanceLoader._load_excel(file_path)
            elif file_path.suffix == ".json":
                return ImpedanceLoader._load_json(file_path)
            else:
                raise DataLoadError(f"Unsupported file format: {file_path.suffix}")
        except Exception as e:
            raise DataLoadError(f"Error loading {file_path}: {str(e)}")

    @staticmethod
    def _load_txt(file_path: Path) -> ImpedanceData:
        """Load from text file, auto-detect delimiter"""
        try:
            # Try reading first line to check if it's a header
            with open(file_path) as f:
                first_line = f.readline().strip()

            # If first line contains letters, it's likely a header - skip it
            if any(c.isalpha() for c in first_line):
                data = pd.read_csv(
                    file_path,
                    sep=None,
                    engine="python",
                    names=["freq", "zreal", "zimag"],
                    skiprows=1,
                )
            else:
                data = pd.read_csv(
                    file_path,
                    sep=None,
                    engine="python",
                    names=["freq", "zreal", "zimag"],
                )

            return ImpedanceLoader._process_dataframe(data)
        except Exception as e:
            raise DataLoadError(f"Failed to load data: {str(e)}")

    @staticmethod
    def _load_csv(file_path: Path) -> ImpedanceData:
        """Load from CSV file"""
        try:
            # Try reading first line to check if it's a header
            with open(file_path) as f:
                first_line = f.readline().strip()

            # If first line contains letters, it's likely a header - skip it
            if any(c.isalpha() for c in first_line):
                data = pd.read_csv(
                    file_path,
                    sep=None,
                    engine="python",
                    names=["freq", "zreal", "zimag"],
                    skiprows=1,
                )
            else:
                data = pd.read_csv(
                    file_path,
                    sep=None,
                    engine="python",
                    names=["freq", "zreal", "zimag"],
                )

            return ImpedanceLoader._process_dataframe(data)
        except Exception as e:
            raise DataLoadError(f"Failed to load CSV: {str(e)}")

    @staticmethod
    def _load_excel(file_path: Path) -> ImpedanceData:
        """Load from Excel file"""
        try:
            # Read first row to check if it's a header
            first_row = pd.read_excel(file_path, nrows=1)

            # If first row contains any text, treat it as header and skip
            if first_row.iloc[0].astype(str).str.contains("[a-zA-Z]").any():
                data = pd.read_excel(
                    file_path, names=["freq", "zreal", "zimag"], skiprows=1
                )
            else:
                data = pd.read_excel(file_path, names=["freq", "zreal", "zimag"])

            return ImpedanceLoader._process_dataframe(data)
        except Exception as e:
            raise DataLoadError(f"Failed to load Excel: {str(e)}")

    @staticmethod
    def _load_json(file_path: Path) -> ImpedanceData:
        """Load from JSON file"""
        try:
            # For JSON, we'll always use our column names since JSON should have a schema
            data = pd.read_json(file_path)

            # Ensure we only take the first 3 columns
            data = data.iloc[:, :3]
            data.columns = ["freq", "zreal", "zimag"]

            return ImpedanceLoader._process_dataframe(data)
        except Exception as e:
            raise DataLoadError(f"Failed to load JSON: {str(e)}")

    @staticmethod
    def _process_dataframe(df: pd.DataFrame) -> ImpedanceData:
        """Process dataframe into ImpedanceData with standard headers"""
        try:
            frequency = df.iloc[:, 0].astype(float).to_numpy()
            real = df.iloc[:, 1].astype(float).to_numpy()
            imaginary = df.iloc[:, 2].astype(float).to_numpy()

            # Ensure descending frequency order
            if not np.all(np.diff(frequency) < 0):
                idx = np.argsort(frequency)[::-1]
                frequency = frequency[idx]
                real = real[idx]
                imaginary = imaginary[idx]

            return ImpedanceData(frequency=frequency, real=real, imaginary=imaginary)
        except Exception as e:
            raise DataLoadError(f"Failed to process data: {str(e)}")
