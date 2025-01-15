# src/fitters/linkk.py
import numpy as np
import logging
from typing import Optional, Tuple
from impedance.validation import linKK
from ..core.models import ImpedanceData, LinKKResult


class LinKKFitter:
    def __init__(self, data: ImpedanceData):
        self.freq = data.frequency
        self.Z = data.real + 1j * data.imaginary
        self.logger = logging.getLogger(__name__)

    def fit(self, c: float = 0.85, max_M: int = 100) -> Optional[LinKKResult]:
        """Perform Lin-KK analysis

        Args:
            c: cutoff ratio for determining optimal number of RC elements
            max_M: maximum number of RC elements to try

        Returns:
            LinKKResult object containing fit results
        """
        try:
            self.logger.info("Starting Lin-KK analysis")
            M, mu, Z_fit, res_real, res_imag = linKK(
                f=self.freq,
                Z=self.Z,
                c=c,
                max_M=max_M,
                fit_type="complex",
                add_cap=True,
            )

            # Calculate residuals and quality metrics
            residuals = np.abs((self.Z - Z_fit) / self.Z)
            max_residual = np.max(residuals)
            mean_residual = np.mean(residuals)

            self.logger.info("Lin-KK Results:")
            self.logger.info(f"M (RC elements): {M}")
            self.logger.info(f"mu parameter: {mu:.6f}")
            self.logger.info(f"Mean residual: {mean_residual:.6e}")
            self.logger.info(f"Max residual: {max_residual:.6e}")

            result = LinKKResult(
                M=M,
                mu=float(mu),
                Z_fit=Z_fit,
                residuals_real=res_real,
                residuals_imag=res_imag,
                max_residual=float(max_residual),
                mean_residual=float(mean_residual),
            )

            self.logger.debug(f"Created LinKKResult object: {result}")
            return result

        except Exception as e:
            self.logger.error(f"Lin-KK fitting failed: {str(e)}", exc_info=True)
            return None
