# src/agent/tools/fitter_tools.py
from typing import Optional
import traceback
import numpy as np
import jax.numpy as jnp
from ...core.models import (
    ImpedanceData,
    FitResult,
    DRTResult,
    LinKKResult,  # Add this import
)
from ...fitters.ecm import ECMFitter
from ...fitters.drt import DRTFitter
from ...fitters.linkk import LinKKFitter  # Add this import too


class FitterTools:
    """Tools for ECM and DRT fitting with error handling"""

    def run_ecm_fit(self, data: ImpedanceData, **kwargs) -> Optional[FitResult]:
        """Run ECM fitting"""
        try:
            print("ECM fitting arguments:", kwargs)
            model_code = kwargs.get("model_code")
            variables = kwargs.get("variables", [])
            weighting = kwargs.get("weighting", "modulus")

            if not model_code or not variables:
                raise ValueError(
                    "Missing required parameters: model_code and variables"
                )

            # Extract parameters and convert to numpy arrays
            p0 = np.array([var["initialValue"] for var in variables])
            lb = np.array([var["lowerBound"] for var in variables])
            ub = np.array([var["upperBound"] for var in variables])

            # Create namespace with required imports
            namespace = {
                "jnp": jnp,
                "np": jnp,
            }

            # Execute model code in namespace
            exec(model_code, namespace)
            model_func = list(namespace.values())[-1]

            # Create and run fitter
            fitter = ECMFitter(
                model_func=model_func,
                p0=p0,
                freq=data.frequency,
                impedance_data=data,
                lb=lb,
                ub=ub,
                param_info=variables,
                weighting=weighting,
            )

            result = fitter.fit()
            if result is None:
                raise ValueError("ECM fitting failed to produce a result")

            return result

        except Exception as e:
            print(f"ECM fitting failed: {str(e)}")
            import traceback

            traceback.print_exc()
            return None

    def run_drt_fit(self, data: ImpedanceData, **kwargs) -> Optional[DRTResult]:
        try:
            lambda_t = kwargs.get("lambda_t", 1e-14)
            lambda_pg = kwargs.get("lambda_pg", 0.01)
            mode = kwargs.get("mode", "real")

            # Get data in original order (it's already in descending frequency)
            freqs = np.array(data.frequency)  # Already in descending order
            omega = 2 * np.pi * freqs
            zre = np.array(data.real)
            zim = np.abs(np.array(data.imaginary))  # Make positive like original

            print("\nData before shifting:")
            print(f"Frequency: {freqs[:5]} ... {freqs[-5:]}")
            print(f"Real: {zre[:5]} ... {zre[-5:]}")
            print(f"Imag: {zim[:5]} ... {zim[-5:]}")

            # Shift by high frequency point like original
            zre = zre - zre[0]

            print("\nAfter shifting:")
            print(f"Real range: {zre[0]} to {zre[-1]}")
            print(f"Expected Rpol = {zre[-1] - zre[0]}")  # Should be positive now

            fitter = DRTFitter(
                zexp_re=zre,
                zexp_im=zim,
                omg=omega,
                lam_t0=lambda_t,
                lam_pg0=lambda_pg,
                lower_bounds=jnp.array([1e-15, 1e-15]),
                upper_bounds=jnp.array([1e15, 1e15]),
                mode=mode,
            )

            return fitter.fit()

        except Exception as e:
            print(f"DRT fitting failed: {str(e)}")
            traceback.print_exc()
            return None

    def run_linkk_fit(self, data: ImpedanceData, **kwargs) -> Optional[LinKKResult]:
        """Run Lin-KK validation"""
        try:
            c = kwargs.get("c", 0.85)
            max_M = kwargs.get("max_M", 100)

            fitter = LinKKFitter(data)
            return fitter.fit(c=c, max_M=max_M)

        except Exception as e:
            print(f"Lin-KK validation failed: {str(e)}")
            import traceback

            traceback.print_exc()
            return None
